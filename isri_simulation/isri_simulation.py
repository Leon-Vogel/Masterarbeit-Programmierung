
import warnings
import pickle
warnings.simplefilter(action='ignore', category=UserWarning)
import copy
import time
from datetime import datetime, timedelta
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyodbc
import simpy


PRECISION = 1e-10 #Simpy Erzeugt Floating Point Fehler -> Alles kleiner als dieser Wert ist = 0
DIGITS = 5 # Runde auf so viele Nachkommastellen Alternative zu PRECISION Ansatz
DIVIDER = float(10**DIGITS)
DEBUG = False
MAX_ITERS = 200

def fast_round(x: float):
    """Schnelleres Runden: https://stackoverflow.com/questions/44920655/python-round-too-slow-faster-way-to-reduce-precision"""
    r = int(x * DIVIDER + 0.5)/DIVIDER
    return r

class Siminfo:
    'Lädt Informationen für Simulationen aus DB und macht preprocessing'
    def __init__(self, conn_str: str, start_time: datetime, print_options: dict, print_subjects: list) -> None:
        self.conn_str = conn_str
        self.start_time = start_time
        self.print_options = print_options
        self.print_subjects = print_subjects
         # Initialisiere alle Linien
        con = pyodbc.connect(conn_str)
        self.stations = pd.read_sql("SELECT * FROM [dbo].Station", con)
        substations = pd.read_sql("SELECT * FROM [dbo].Substation", con)

        # Definiere Schichten
        shifts = pd.read_sql(f"SELECT * FROM Shift", con).sort_values(by=["ShiftDay", "ShiftStart"])
        shifts.ShiftStart = pd.to_datetime(shifts.ShiftStart, format='%H:%M:%S.0000000')
        shifts.ShiftEnd = pd.to_datetime(shifts.ShiftEnd, format='%H:%M:%S.0000000')
        start_hours = shifts.ShiftStart.dt.hour.tolist()
        start_minutes = shifts.ShiftStart.dt.minute.tolist()

        end_hours = shifts.ShiftEnd.dt.hour.tolist()
        end_minutes = shifts.ShiftEnd.dt.minute.tolist()
        shiftday = shifts.ShiftDay.astype(int).tolist()

        shift_starts = [datetime(start_time.year, start_time.month, start_time.day, start_hours[i], start_minutes[i])
                         + timedelta(days=(shiftday[i] - start_time.weekday())) for i in range(len(shifts))]
        shift_starts = [start + timedelta(weeks=1) if start < start_time else start for start in shift_starts]
        shift_starts = sorted(shift_starts)
        self.shift_starts = [(start - start_time).total_seconds() for start in shift_starts]

        shift_ends = [datetime(start_time.year, start_time.month, start_time.day, end_hours[i], end_minutes[i])
                         + timedelta(days=(shiftday[i] - start_time.weekday())) for i in range(len(shifts))]
        shift_ends = [end + timedelta(weeks=1) if end < start_time else end for end in shift_ends]
        shift_ends = sorted(shift_ends)
        self.shift_ends = [(start - start_time).total_seconds() for start in shift_ends]

        # Evtl. muss erst auf Schichtbeginn gewartet werden
        if shift_ends[0] <= shift_starts[0]:
            wait_time = None
            self.wait_for_first_shift = False
        else:
            wait_time = shift_starts[0]
            self.wait_for_first_shift = True
        self.wait_time = wait_time

        # Info für alle Linien und OPs
        self.station_info = {}
        self.clockrates = {}
        self.error_infos = {}
        self.maintenance_starts = {}
        self.maintenance_ends = {}
        self.n_machines = 0
        for idx, station in self.stations.iterrows():
            maintenance = pd.read_sql(f"""SELECT * FROM Pause 
                WHERE StationID = {station.StationID}
                AND MaintenanceEnd > '{start_time.strftime('%Y-%m-%d %H:%M')}'
                ORDER BY MaintenanceStart""", con)

            self.maintenance_starts[station.StationID] = maintenance.MaintenanceStart.tolist()
            self.maintenance_ends[station.StationID] = maintenance.MaintenanceEnd.tolist()

            # Initialisiere die OPs
            this_substations = substations[substations.StationID == station.StationID].sort_values(by="Sequence", ascending=False)
            self.station_info[station.StationID] = this_substations
            self.n_machines = len(this_substations)
            for idx, substation in this_substations.iterrows():
                clockrate_query = f"""
                    SELECT
                        avgTime
                    FROM
                        SubstationTaskInfo sti
                    LEFT JOIN
                        Substation ss on ss.SubstationID = sti.SubstationID
                    WHERE
                        ss.SubstationID = {substation.SubstationID}
                    """
                clockrate = pd.read_sql(clockrate_query, con).values[0][0]
                self.clockrates[f'{station.StationID}_{substation.SubstationID}'] = clockrate
                errors = pd.read_sql(f"SELECT * FROM SubstationError WHERE SubstationID = {substation.SubstationID}", con)
                infos = [[errors.Prob.iloc[i], errors.Seconds.iloc[i]] for i in range(len(errors))]
                error_info = dict(zip(errors.ErrorCode, infos))
                self.error_infos[f'{station.StationID}_{substation.SubstationID}'] = error_info

        con.close()

class Simulation:
    def __init__(self, siminfo: Siminfo):
        """
        simrun_id: id für jeden Simulationslauf (falls Ergebnisse gespeichert werden sollen zur weiteren Analyse)
        conn_str: jdbc connection string zu einer Datenbank mit entsprechendem Schema
        """
        
        self.env = simpy.Environment()
        self.print_options = siminfo.print_options
        self.print_subjects = siminfo.print_subjects
        self.log = Log(self, print_categories=siminfo.print_options, print_subjects=siminfo.print_subjects)
        self.stations = []
        self.run_proc = None
        self.products = []
        self.shift_starts = siminfo.shift_starts
        self.shift_ends = siminfo.shift_ends
        self.wait_time = siminfo.wait_time
        # self.wait_for_first_shift = siminfo.wait_for_first_shift
        self.maintenance_starts = siminfo.maintenance_starts
        self.maintenance_ends = siminfo.maintenance_ends
    
        
        for idx, station in siminfo.stations.iterrows():
            substation_list = []
            for idx, substation in siminfo.station_info[station.StationID].iterrows():
                error_info = siminfo.error_infos[f'{station.StationID}_{substation.SubstationID}']
                clockrate = siminfo.clockrates[f'{station.StationID}_{substation.SubstationID}']
                if len(substation_list) == 0:
                    substation_obj = Substation(substation.SubstationID, next_substation=None, station=None, sequence=substation.Sequence, time_in_station=clockrate,
                                                sim=self, error_prob=error_info)
                else:
                    substation_obj = Substation(substation.SubstationID, next_substation=substation_list[0], station=None, sequence=substation.Sequence, time_in_station=clockrate,
                                                sim=self, error_prob=error_info)
                substation_list.insert(0, substation_obj)

            station_obj = Station(station.StationID, station.Name, station.currSpeed, substation_list, self)
            self.stations.append(station_obj)
            for sub in substation_list:
                sub.station = station_obj
    
    def run(self, plan: dict, start_time: datetime, max_time: int = 100000, finish_cond=None):
        """
        Startet die Simulation
        """
        for idx, station in enumerate(self.stations):
            self.products = self.products + plan[station.station_id]
            
            if station.station_id in self.maintenance_starts.keys():
                maintenance_starts = self.maintenance_starts[station.station_id]
                maintenance_ends = self.maintenance_ends[station.station_id]
                if len(maintenance_starts) > 0:
                    if maintenance_starts[0] < start_time and maintenance_ends[0] > start_time:
                        # Wir sind schon in der Wartung -> Wartung startet jetzt
                        maintenance_starts[0] = start_time
                    maintenance_starts = [(t - start_time).total_seconds() for t in maintenance_starts]
                    maintenance_ends = [(t - start_time).total_seconds() for t in maintenance_ends]
                else:
                    maintenance_starts = None
                    maintenance_ends = None
            else:
                maintenance_ends = None
                maintenance_starts = None
                
            station.run_proc = self.env.process(station.run(prod_plan=plan[station.station_id],
                                                            shift_starts=self.shift_starts,
                                                            shift_ends=self.shift_ends,
                                                            delay_start= idx*0.0000001,
                                                            maintenance_starts=maintenance_starts,
                                                            maintenance_ends=maintenance_ends))

        if finish_cond is None:
            finish_proc = self.env.process(self.finish_condition(self.env, max_time))
        else:
            finish_proc = self.env.process(finish_cond)
        self.log.start_time = start_time
        # self.env.process(self.shift_process(self.start_time, self.wait_for_first_shift, self.shift_starts, self.shift_ends, plan))
        self.log.print_message(f"Starting Simulation", ["General"], ["Sim"])
        
        # self.env.run(until=finish_proc)
        self.env.run(finish_proc)
        self.log.simend = self.env.now
        # print([s.iters for s in self.stations[0].substations])

    def step(self):
        """
        Lässt die Simulation so lange weiterlaufen bis ein neues Produkt gewählt werden muss
        """
        self.env.run(simpy.events.AnyOf(self.env, [s.needs_product for s in self.stations]))
        station_calling = [s for s in self.stations if s.needs_product.processed]
        station_calling[0].needs_product = self.env.event()
        return station_calling[0]

        
    def shift_process(self, time: datetime, wait_for_first: bool, shift_starts: list, shift_ends: list, prod_plan: dict):
        """
        Steuert die einzelnen Schichten und Pausen dazwischen. Stationen werden außerhalb der Schichten interrupted.
        """
        self.log.print_message(f"Starting Simulation", ["General"])
        shift_ind = 0
        if wait_for_first:
            wait_time = (shift_starts[0] - time).total_seconds()
            self.log.print_message(f"Waiting {wait_time} for first shift", ["General"])
            yield self.env.timeout(wait_time)
        
        for station in self.stations:
            station.run_proc = self.env.process(station.run(prod_plan = prod_plan[station.station_id]))

        while True:
            shift_end = shift_ends[shift_ind]
            time_until_end = (shift_end - (time + timedelta(seconds=self.env.now))).total_seconds()
            yield self.env.timeout(time_until_end)
            next_shift_start = shift_starts[shift_ind + 1]
            pause_time = (next_shift_start - shift_end).total_seconds()
            self.log.print_message(f"Shift ended next shift in {pause_time}", ["General"])
            for station in self.stations:
                station.interrupt(pause_time)
            shift_ind += 1

    def finish_condition(self, env: simpy.Environment, max_time):
        """
        Stoppt die Simulation wenn alle Stationen ihren Produktionsplan fertig haben
        """
        yield simpy.events.AllOf(env, [s.finish_event for s in self.stations]) | env.timeout(max_time)
        self.log.print_message("Finished Simulation", ["General"], ["Sim"])

    def reset(self):
        """
        Reset simulation after run
        """
        self.products = []
        self.run_proc = None
        self.env = simpy.Environment()
        self.log = Log(sim=self, print_categories=self.print_options, print_subjects=self.print_subjects)
        for s in self.stations:
            s.finish_event = self.env.event()
            s.run_proc = None
            s.env = self.env
            for ss in s.substations:
                ss.processes = []
                ss.env = self.env
                ss.employee = simpy.Resource(self.env, capacity = 1)

class Log:
    def __init__(self,sim: Simulation,  print_categories: dict=None, print_subjects:list=None) -> None:
        self.error_log = {
            "Station": [],
            "Substation": [],
            "Reason": [],
            "Start": [],
            "End": []
        }

        self.eventlog = {
            "Product": [],
            "Station": [],
            "Substation": [],
            "Entry": [],
            "ReceivedMA": [],
            "FinishedWork": [],
            "ReachedEnd": []
        }

        self.maintenance_log = {
            "Station": [],
            "Start": [],
            "End": []
        }

        self.pause_log = {
            "Station": [],
            "Start": [],
            "End": []
        }

        self.debug_log = {
            'station': [],
            'substation': [],
            'function': [],
            'time': [],
            'info': [],
            'positions': [],
            'worktime': [],
            'update_time': [],
            'total_interrupt_time': []
        }

        if print_categories is None:
            self.print_categories = {
                "General": True,
                "Production": False,
                "Warnings": True
            }
        else:
            self.print_categories = print_categories
        
        self.print_subjects = print_subjects
        self.sim = sim
        self.start_time = None
        self.simend = None
        self.last_error_start = {}
        self.last_error_end = {}
    
    def log_move(self, product, station, substation, start, ma_time, worktime, reached_end) -> None:
        self.eventlog["Product"].append(product)
        self.eventlog["Station"].append(station)
        self.eventlog["Substation"].append(substation)
        self.eventlog["Entry"].append(start)
        self.eventlog["ReceivedMA"].append(ma_time)
        self.eventlog["FinishedWork"].append(worktime)
        self.eventlog["ReachedEnd"].append(reached_end)

    def log_error(self, station, substation, reason, start, end) -> None:
        self.error_log["Station"].append(station)
        self.error_log["Substation"].append(substation)
        self.error_log["Reason"].append(reason)
        self.error_log["Start"].append(start)
        self.error_log["End"].append(end)
        self.last_error_start[station] = start
        self.last_error_end[station] = end

    def log_maintenance(self, station, start, end) -> None:
        self.maintenance_log["Station"].append(station)
        self.maintenance_log["Start"].append(start)
        self.maintenance_log["End"].append(end)

    def log_pause(self, station, start, end) -> None:
        self.pause_log["Station"].append(station)
        self.pause_log["Start"].append(start)
        self.pause_log["End"].append(end)
    
    def print_message(self, message: str, categories: list, subject: list) -> None:
        categories_options = [self.print_categories[s] for s in categories]
        if self.print_subjects is not None:
            subject_relevant = np.any([s in self.print_subjects for s in subject])
        else:
            subject_relevant = True
        if np.any(categories_options) or subject_relevant:
            print(f"{self.sim.env.now:.2f} {message}")
    
    def log_debug_info(self, obj, function_name, status):
        self.debug_log['station'].append(obj.station.name)
        self.debug_log['substation'].append(obj.substation_id)
        self.debug_log['function'].append(function_name)
        self.debug_log['time'].append(self.sim.env.now)
        self.debug_log['info'].append(status)
        self.debug_log['positions'].append(copy.deepcopy(obj.positions))
        self.debug_log['worktime'].append(copy.deepcopy(obj.worktime))
        self.debug_log['update_time'].append(copy.deepcopy(obj.last_update_time))
        self.debug_log['total_interrupt_time'].append(copy.deepcopy(obj.total_interrupt_time_since_positions_update))


    def save_to_csv(self, path: str):
        eventlogs = pd.DataFrame(self.eventlog)
        eventlogs['WaitTime'] = eventlogs.ReceivedMA - eventlogs.Entry
        eventlogs['Worktime'] = eventlogs.FinishedWork - eventlogs.ReceivedMA
        eventlogs['TransportTime'] = eventlogs.ReachedEnd - eventlogs.FinishedWork
        eventlogs['TotalTime'] = eventlogs.WaitTime + eventlogs.Worktime + eventlogs.TransportTime

        error = pd.DataFrame(self.error_log)
        eventlogs.to_csv(path+"eventlog.csv", sep=";")
        error.to_csv(path+"errors.csv", sep=";")

    def deadline_goal(self):
        simend = self.start_time + timedelta(seconds=self.sim.env.now)
        deadlines_gaps = [(p.finish_time - p.due_date).total_seconds() if p.finish_time is not None else (simend - p.due_date).total_seconds() for p in self.sim.products]
        return deadlines_gaps
    
    def workload_goal(self, function='minvar'):
        workloads = []
        data = pd.DataFrame(self.eventlog)
        for station in self.sim.stations:
            for substation in station.substations:
                substation_data = data[data.Substation == substation.substation_id].copy()
                substation_data['LastFinish'] = substation_data.FinishedWork.shift(1, fill_value=0)
                available_time = substation_data.ReceivedMA - substation_data.LastFinish
                available_time = available_time.tolist()
                available_time[0] = substation.time_in_station
                available_time[-1] = substation.time_in_station # Beim ersten und letzten Produkt kann kein sinnvoller Wert bestimmt werden
                # if np.any(np.array(available_time, dtype=np.float16) < 0):
                #     print("?")
                workloads.append(available_time)
        return np.array(workloads)
    
    def plot_station_hist(self):
        fig, ax = plt.subplots()
        df_maintenance = pd.DataFrame(self.maintenance_log)
        df_maintenance["Duration"] = df_maintenance.End - df_maintenance.Start
        df_pause = pd.DataFrame(self.pause_log)
        df_pause["Duration"] = df_pause.End - df_pause.Start
        df_error = pd.DataFrame(self.error_log)
        df_error["Duration"] = df_error.End - df_error.Start
        df_eventlog = pd.DataFrame(self.eventlog)
        y_start = 0
        for station in self.sim.stations:
            facecolors = []
            name = station.name

            station_maintenance = df_maintenance[df_maintenance.Station == name]
            maintenance_bars_x = list(zip(station_maintenance.Start, station_maintenance.Duration))
            facecolors = facecolors + ['tab:orange'] * len(maintenance_bars_x)

            station_pause = df_pause[df_pause.Station == name]
            pause_bars_x = list(zip(station_pause.Start, station_pause.Duration))
            facecolors = facecolors + ['tab:green'] * len(pause_bars_x)

            station_error = df_error[df_error.Station == name]
            error_bars_x = list(zip(station_error.Start, station_error.Duration))
            facecolors = facecolors + ['tab:red'] * len(error_bars_x)

            bars_x = maintenance_bars_x + pause_bars_x + error_bars_x
            bars_y = (y_start, 9)
            end_work = df_eventlog[df_eventlog.Station == name].ReachedEnd.iloc[-1]
            rectangle = patches.Rectangle((0, y_start), end_work, 9, facecolor='grey', alpha=0.3)
            ax.add_patch(copy.deepcopy(rectangle))
            ax.broken_barh(bars_x, bars_y, facecolors=facecolors)
            y_start += 10
        ax.set_xlabel("seconds since simstart")
        ax.set_xticks(np.linspace(0, self.simend, 10))
        ax.set_yticks(np.arange(5, len(self.sim.stations) * 10 + 1, 10), labels=[s.name for s in self.sim.stations])
        orange_patch = patches.Patch(color='tab:orange', label='Maintenance')
        green_patch = patches.Patch(color='tab:green', label='Shift-Ended')
        red_patch = patches.Patch(color='tab:red', label='Error')
        grey_patch = patches.Patch(color='lightgray', label="Working")
        plt.legend(handles=[orange_patch, green_patch, red_patch, grey_patch])
        plt.show()

    def plot_employee_hist(self):
        fig, ax = plt.subplots()
        df_eventlog = pd.DataFrame(self.eventlog)
        for station in self.sim.stations:
            for substation in station.substations:
                substation_data = df_eventlog[df_eventlog.Substation == substation.substation_id].copy()
                substation_data['LastFinish'] = substation_data.FinishedWork.shift(1, fill_value=0)
                available_time = substation_data.ReceivedMA - substation_data.LastFinish
                available_time = available_time.tolist()
                available_time[0] = 0
                available_time[-1] = 0 # Beim ersten und letzten Produkt kann kein sinnvoller Wert bestimmt werden
                # if np.any(np.array(available_time, dtype=np.float16) < 0):
                #     print("?")

        plt.show()   

class Station:
    def __init__(self, station_id: int, name: str, speed: float, substations: list, sim: Simulation) -> None:
        self.station_id = station_id
        self.name = name
        self.speed = speed
        self.substations = substations
        self.start_substation = substations[0]
        self.sim = sim
        self.env = sim.env
        self.run_proc = None
        self.finish_event = self.env.event()
        self.interrupted = False
        self.shift_index = None
        self.start_shift = None
        self.maintenance_index = 0
        self.shift_starts = None
        self.shift_ends = None
        self.maintenance_starts = None
        self.maintenance_ends = None
        self.needs_product = self.env.event()
    
    def run(self, prod_plan: list, shift_starts: list, shift_ends: list, delay_start: int = None, maintenance_starts: list = None, maintenance_ends: list = None):
        """
        Puts a new product on the start of the line every self.speed seconds.
        """
        planned_production_time = sum([s.time_in_station for s in self.substations])
        time_to_next_prod = self.speed
        interrupt_time = delay_start
        while True:
            try:
                if interrupt_time > 0:
                    wait_for_interrupt_start = self.env.now
                    yield self.env.timeout(interrupt_time)
                    interrupt_time = 0
                if len(prod_plan) > 0:
                    self.needs_product.succeed()
                    next_prod = prod_plan.pop(0)
                    self.substations[0].produce(next_prod)
                else:
                    break
                yield self.env.timeout(time_to_next_prod)
            except simpy.Interrupt as interrupt:
                interrupt_start = self.env.now
                if interrupt_time > 0:
                    time_waited = interrupt_start - wait_for_interrupt_start
                    interrupt_time = interrupt_time - time_waited + interrupt.cause
                else:
                    interrupt_time = interrupt.cause
        
        # Wait for Products to finish
        last_prod_finished = False
        while not last_prod_finished:
            try:
                yield next_prod.finished
                last_prod_finished = True
            except simpy.Interrupt as interrupt:
                pass
        self.run_proc = None
        self.finish_event.succeed()
    
    def interrupt(self, time):
        """
        Interrupts all processes within the station for time seconds
        """
        self.sim.log.print_message(f"{self.name} has been interrupted for {time:.2f}", ["General"], [f"Station {self.name}"])
        self.interrupted = True
        for substation in self.substations:
            for proc in substation.processes:
                if proc.is_alive and proc is not self.env.active_process:
                    proc.interrupt(time)
        if self.run_proc is not None:
            self.run_proc.interrupt(time)
        self.env.process(self._interrupt(time))
    
    def _interrupt(self, time):
        """
        Simpy Process for interruption time
        """
        yield self.env.timeout(time)
        self.interrupted = False

class Product:
    def __init__(self, product_id: int, times: list, due_date: datetime, sim: Simulation):
        self.product_id = product_id
        self.times = times
        self.due_date = due_date
        self.finish_time = None
        self.sim = sim
        self.location = None
        self.finished = self.sim.env.event()

    def finish_production(self):
        self.finished.succeed()
        self.finish_time = self.sim.log.start_time + timedelta(seconds=self.sim.env.now)

class Substation:
    def __init__(self, substation_id: int, next_substation, station: Station, sequence: int, time_in_station: float,
                sim: Simulation, error_prob: dict) -> None:
        self.substation_id = substation_id
        self.station = station
        self.time_in_station = time_in_station
        self.sequence = sequence
        self.state = 'working'
        self.sim = sim
        self.env = sim.env
        self.next_substation = next_substation
        self.error_prob = error_prob # Dict mit error_id: [probability, fail_time]
        self.positions = {}
        self.last_update_time = {}
        self.worktime = {}
        self.get_product_event = self.env.event()
        self.working_on_id = None
        self.last_working_update_time = 0
        self.interrupt_time = 0
        self.total_interrupt_time_since_positions_update = 0
        self.products = {}
        self.log_rows = {}
        self.interrupt_started_at = 0
        self.iters = 0
        
        # Work process runs forever
        proc = self.env.process(self.work_process())
        self.processes = [proc]
        
    def produce(self, product: Product) -> None:
        self.positions[product.product_id] = 0
        self.last_update_time[product.product_id] = self.env.now
        self.worktime[product.product_id] = max(product.times[self.sequence], 1) # Jedes Produkt wird mind. 1 Sekunde bearbeitet
        self.products[product.product_id] = product
        self.log_rows[product.product_id] = {'Entry': self.env.now, 'StartWork': None,
                                              'Worktime': product.times[self.sequence], 'FinishedWork': None}
        self.get_product_event.succeed()
        self.get_product_event = self.env.event()
        # Substation ist aktuell leer
        if self.working_on_id is None:
            self.working_on_id = product.product_id
            self.last_working_update_time = self.env.now
            self.log_rows[product.product_id]['StartWork'] = self.env.now

    def work_process(self):
        while self.iters < MAX_ITERS:
            try:
                self.iters += 1
                # Mögliche Unterbrechung abwarten
                if self.interrupt_time > 0:
                    self.wait_for_interrupt_t = self.env.now
                    self.state = 'interrupted'
                    yield self.env.timeout(self.interrupt_time)
                    self.interrupt_time = 0

                self._update_worktime()
                self._update_positions()
                self.state = 'working'

                # Warte auf das nächste Event
                next_event, next_event_time = self._get_next_event()
                yield self.env.timeout(next_event_time) | self.get_product_event

            except simpy.Interrupt as interrupt:
                # Unterbrechung
                self.interrupt_time = interrupt.cause
                if self.state == 'interrupted':
                    self._update_worktime()
                    self._update_positions()
                    self.total_interrupt_time_since_positions_update = interrupt.cause
                else:
                    self.total_interrupt_time_since_positions_update = interrupt.cause
                self.interrupt_started_at = self.env.now
    
    def _update_positions(self) -> None:
        if DEBUG:
            self.sim.log.log_debug_info(self, '_update_positions', 'start')
        time_now = self.env.now
        interruption = False
        for p_id, pos in self._dictionary_iterator(self.positions):
            time_gone = time_now - self.last_update_time[p_id]

            # Falls Produkt in der selben Sekunde eintrifft wie Unterbrechung vorbei 
            # (Produkt war Grund für unterbrechung und ist jetzt an der nächsten Station)
            if time_gone == 0:
                time_running = 0
            else:
                last_interrupt_time = self._get_interrupt_time_since_update(p_id)
                time_running = fast_round(time_gone - last_interrupt_time)

            if time_running < 0:
                raise RuntimeError(f'Linie war {self.total_interrupt_time_since_positions_update} unterbrochen, es sind aber nur {time_gone} Sekunden vergangen')
            
            pos_new = fast_round(pos + time_running)

            # Produkt hat das Ende erreicht
            if pos_new >= self.time_in_station:
                # Fertig mit arbeiten?
                if self.worktime[p_id] <= PRECISION:
                    finished = True
                else:
                    finished = False
                if finished:
                    # Fertig - loggen, löschen
                    log_data = self.log_rows[p_id]
                    self.sim.log.log_move(p_id, self.station.name, self.substation_id, log_data['Entry'], log_data['StartWork'], 
                                        log_data['FinishedWork'], self.env.now)
                    if self.next_substation is not None:
                        self.next_substation.produce(self.products[p_id])
                    else:
                        self.products[p_id].finish_production()
                    del self.positions[p_id]
                    del self.worktime[p_id]
                    del self.products[p_id]
                    del self.last_update_time[p_id]
                    if p_id == self.working_on_id:
                        self.working_on_id = self._get_next_prod_with_positive_worktime()
                else:
                    # Nicht fertig - Linie unterbrechen
                    worktime_left = self.worktime[p_id]
                    self.positions[p_id] = self.time_in_station
                    self.last_update_time[p_id] = time_now
                    self.sim.log.log_error(self.station.name, self.substation_id, f'Not finished {p_id} in Time at {self.substation_id}', self.env.now, self.env.now + worktime_left)
                    self.station.interrupt(worktime_left)
                    interruption = True
            # Produkt befindet sich noch in der Linie
            else:
                self.positions[p_id] = fast_round(pos_new)
                self.last_update_time[p_id] = time_now
        
        # Falls es zu einer Unterbrechung kam
        if interruption:
            self.interrupt_time = worktime_left
            self.total_interrupt_time_since_positions_update = worktime_left
            self.interrupt_started_at = self.env.now
        else:
            self.total_interrupt_time_since_positions_update = 0

        if DEBUG:
            self.sim.log.log_debug_info(self, '_update_positions', 'end')
    
    def _update_worktime(self) -> None:
        if DEBUG:
            self.sim.log.log_debug_info(self, '_update_worktime', 'start')
        time_now = self.env.now
        time_gone = time_now - self.last_working_update_time
        if len(self.worktime) > 0:
            continue_work = True
        else:
            continue_work = False
        
        while continue_work and self.working_on_id is not None:
            product_worktime = self.worktime[self.working_on_id]
            worktime_left = max(product_worktime - time_gone, 0)
            self.worktime[self.working_on_id] = fast_round(worktime_left)
            # Falls fertig - Floating Point Fehler vermeiden
            if worktime_left <= PRECISION:
                self.log_rows[self.working_on_id]['FinishedWork'] = self.env.now
                self.working_on_id = self._get_next_prod_with_positive_worktime()
                # Es gibt noch ein weiteres Produkt zum Arbeiten
                if self.working_on_id is not None:
                    self.log_rows[self.working_on_id]['StartWork'] = self.env.now
                    # Falls noch Zeit übrig
                    if time_gone > product_worktime:
                            continue_work = True
                            time_gone -= product_worktime
                    else:
                        continue_work = False
            # Produkt ist noch nicht fertig -> es wird an keinem weiteren gearbeitet
            else:
                continue_work = False

        self.last_working_update_time = self.env.now
        if DEBUG:
            self.sim.log.log_debug_info(self, '_update_worktime', 'end')

    def _get_next_prod_with_positive_worktime(self):
        """
        Nächstes Produkt an dem noch gearbeitet werden muss. Das vorderste Produkt wird zuerst bearbeitet
        """
        positive_worktime_prods = []
        positions = []
        for p_id, time in self.worktime.items():
            if time > PRECISION:
                positive_worktime_prods.append(p_id)
                positions.append(self.positions[p_id])
        if len(positions) > 0:
            arg_max = np.argmax(positions)
            return positive_worktime_prods[arg_max]
        else:
            return None

    def _get_next_product(self, include_current_working: bool=True) -> int:
        """
        Vorderstes Produkt auf der Linie
        """
        if not include_current_working:
            positions = {p_id: pos for p_id, pos in self.positions.items()}
        else:
            positions = self.positions
        if len(positions) > 0:
            return max(positions, key=positions.get)
        else:
            return None
    
    def _get_next_event(self) -> (str, int):
        # Ereignisse: Prod erreicht Ende, Prod wurde zuende bearbeitet
        if self.working_on_id is not None:
            next_worktime_end = self.worktime[self.working_on_id]
        else:
            next_worktime_end = 1e15
        
        first_prod_on_line = self._get_next_product()
        if first_prod_on_line is not None:
            next_transport_end = fast_round(self.time_in_station - self.positions[first_prod_on_line])
        else:
            next_transport_end = 1e15

        if next_worktime_end < next_transport_end:
            event_type = 'work_end'
            event_time = next_worktime_end
        elif next_transport_end < next_worktime_end:
            event_type = 'reached_end'
            event_time = next_transport_end
        else:
            event_type = 'both'
            event_time = next_transport_end
        
        if self.station.name =='LinieA' and self.sequence==4:
            x = 6
        return event_type, event_time

    def _get_interrupt_time_since_update(self, p_id) -> int:
        try:
            interrupt_start = self.sim.log.last_error_start[self.station.name]
            interrupt_end = self.sim.log.last_error_end[self.station.name]
            update_time = self.last_update_time[p_id]
            if update_time > interrupt_end:
                return 0
            elif update_time <= interrupt_end and interrupt_start < self.env.now:
                if update_time < interrupt_start:
                    return interrupt_end - interrupt_start
                else:
                    return (interrupt_end - interrupt_start) - (update_time - interrupt_start)
            else:
                return 0
        except KeyError:
            return 0

    @staticmethod
    def _dictionary_iterator(dict):
        return list(dict.items())
