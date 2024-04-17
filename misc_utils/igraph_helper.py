import igraph as ig
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from misc_utils.copy_helper import fast_deepcopy
from node2vec import Node2Vec
import torch
from scipy.spatial.distance import pdist, squareform

def plot_disjunctive_graph(graph, path):
    labels = [str(graph.vs()[i]['job']) + " " + str(graph.vs()[i]['resource']) + 
              " " + str(graph.vs()[i]['op']) 
              if graph.vs()[i]['op'] is not None else 'START' for i in range(graph.vcount())]
    edge_colors = ['black' if e['edge_type'] == 1 else 'green' for e in graph.es()]
    ig.plot(graph, path, vertex_label=labels, edge_color=edge_colors)


def _get_node_from_graph(graph, job, operation):
    selection = [vertex for vertex in graph.vs() if (vertex['job'] == job) & (vertex['operation'] == operation)]
    if len(selection) == 1:
        return selection[0]
    else:
        raise RuntimeError(f'fNo or multiple nodes for Job {job} and operation {operation} found')


def build_disjunctive_graph(ind: list):
    # Add vertices
    g = ig.Graph(directed=True)
    for idx, gene in enumerate(ind):
        g.add_vertex(idx, job_op=f"{gene.job}_{gene.operation}", job=gene.job, operation=gene.operation, machine=gene.machine)
    
    # Add Job, Machine Edges
    for idx, gene in enumerate(ind):
        machine_successors = [v.index for v in g.vs()[idx + 1:] if v['machine'] == gene.machine]
        job_successors = [(v.index, v['operation']) for v in g.vs() if (v['job'] == gene.job) & (v['operation'] > gene.operation)]
        job_successors.sort(key=lambda x: x[1]) # Sort by operation index

        if len(machine_successors) > 0:
            g.add_edge(idx, machine_successors[0], type=0)
        if len(job_successors) > 0:
            g.add_edge(idx, job_successors[0][0], type=1)
    return g


def _check_is_candidate(graph, gene, target_machine, individual, step):
    "Im Moment wahrscheinlich zu restriktiv - es kann immer nur die nächste OP pro Job verplant werden"
    for i, g in enumerate(individual[step:]):
        if g == gene:
            # Find successors of the gene
            successors = [ind for ind in individual if (ind.job == gene.job) and (ind.operation > gene.operation)]
            for succ in successors:
                # not valid if a job successor is planned at the same machine - everything before step is not fixed yet
                if succ.machine == target_machine and individual.index(succ) < step:
                    return False
                
            # if a job predecessor has only this machine as an option it is also invalid to schedule
            predecessors = [ind for ind in individual if (ind.job == gene.job) and (ind.operation < gene.operation)]
            for pred in predecessors:
                # All predecessors have to be scheduled already - Hier zu restriktiv?
                if not np.all([individual.index(pred) < step]):
                    return False
                # Is ok if pred already scheduled on that machine
                if individual.index(pred) >= step:
                    if len(pred.alt_machines) == 1 and target_machine in pred.alt_machines.keys():
                        return False
            return True
        
    # Gene is already planned
    return False


def _check_if_dag(individual, step, gene, machine):
    ind = individual[:step + 1]
    gene.machine = machine
    ind.append(gene)
    g = build_disjunctive_graph(ind)
    if g.is_dag():
        return True
    else:
        return False


def _calculate_earliest_start(graph, gene, target_machine, machine_nodes, individual):
    """
    Calculates the earliest start time for a job on a given machine.
    """
    # End time of the machine
    machine_end_time = graph.vs[machine_nodes[target_machine]]['end_time']

    # Find predecessors and their end times
    predecessor_end_times = []
    for i, g in enumerate(individual):
        if g.job == gene.job and i < individual.index(gene):
            predecessor_end_time = graph.vs()[i]['end_time']
            if predecessor_end_time is not None:
                predecessor_end_times.append(predecessor_end_time)

    # Earliest start time is the max of machine end time and predecessor end times
    if predecessor_end_times:
        return max(max(predecessor_end_times), machine_end_time)
    else:
        return machine_end_time


def plot_schedule_as_gantt(schedule, ax=None, color_map=None, savepath=None):
    # Initialize a figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax.clear()

    # Color mapping for jobs
    if not color_map:
        color_map = {}
    def get_color(job):
        if job not in color_map:
            # Generate a random color
            color_map[job] = (random.random(), random.random(), random.random())
        return color_map[job]

    # Y-axis labels
    y_labels = list(schedule.keys())
    y_ticks = range(len(y_labels))

    # Plot each operation
    for i, machine in enumerate(y_labels):
        for start_time, end_time, job_id, operation_id in schedule[machine]:
            ax.broken_barh([(start_time, end_time - start_time)], (i-0.4, 0.8), facecolors=get_color(job_id))
            ax.text((start_time + end_time) / 2, i, f"{job_id}, {operation_id}", ha='center', va='center')

    # Set labels and grid
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.grid(True)

    if ax is None:
        plt.show()
    elif savepath is not None:
        plt.savefig(savepath, format='svg')
    
    return color_map


def plot_problem_graph(g, target):
    # Define colors for vertices and edges
    vertex_colors = ["white" if vertex["type"] == 0 else "red" for vertex in g.vs()]
    edge_colors = ["black" if edge["type"] == 0 else "grey" for edge in g.es()]

    # Visual style
    visual_style = {
        "vertex_size": 20,
        "vertex_color": vertex_colors,
        "vertex_label": g.vs["name"],
        "edge_color": edge_colors,
        "bbox": (600, 600),
        "margin": 100
    }

    # Plot the graph
    ig.plot(g, target=target, **visual_style)


def plot_heterogenous_graph_detailed(graph, target, curr_step: int = 0):
    # Define colors for vertices and edges
    vertex_colors = ["white" if vertex["type"] == 0 else "orange" for vertex in graph.vs()]
    edge_colors = []
    for edge in graph.es():
        if edge['type'] == 1:
            edge_colors.append('black')
        elif edge['is_candidate']:
            edge_colors.append('dark orange')
        elif edge['is_chosen']:
            edge_colors.append('dark green')
        elif edge['is_discarded']:
            edge_colors.append('dark red')
        else:
            edge_colors.append('dark magenta')

    # Visual style
    visual_style = {
        "vertex_size": 25,
        "vertex_color": vertex_colors,
        "vertex_label": graph.vs["name"],
        "edge_color": edge_colors,
        "bbox": (600, 600),
        "margin": 100,
        "vertex_label_size": 12
    }

    # Plot the graph
    ig.plot(graph, target=target, **visual_style)

# Helper function to find start and end times from the schedule
def _find_times(job, operation, schedule):
    if schedule is not None:
        for machine_schedule in schedule.values():
            for start_time, end_time, job_id, operation_id in machine_schedule:
                if job_id == job and operation_id == operation:
                    return start_time, end_time
        return -1, -1
    else:
        return None, None


def _get_unique_machines(individual):
    """
    Create a list of all unique machines from the individual data structure.

    Args:
        individual (list): A list of genes, where each gene contains 'machine' and 'alt_machines' attributes.

    Returns:
        list: A list of unique machines.
    """
    machines = set()
    for gene in individual:
        # Add the primary machine
        machines.add(gene.machine)
        # Add all alternative machines
        machines.update(gene.alt_machines.keys())
    
    return list(machines)


def create_edge_graph(g, attributes):
    # Create a new graph for the edge graph
    g_prime = ig.Graph()

    # Add nodes to g_prime, one for each edge in g
    g_prime.add_vertices(g.ecount())

    # Initialize attributes in g_prime
    for attr in attributes:
        g_prime.vs[attr] = [None] * g.ecount()

    # Set attributes for each node in g_prime based on corresponding edge in g
    for e in g.es:
        for attr in attributes:
            g_prime.vs[e.index][attr] = e[attr]

    # A dictionary to store the source and target nodes of each edge
    edge_to_nodes = {e.index: (e.source, e.target) for e in g.es}

    # Add edges to g_prime
    for e1, nodes1 in edge_to_nodes.items():
        for e2, nodes2 in edge_to_nodes.items():
            if e1 < e2:  # To avoid double checking and self-loops
                if (nodes1[0] in nodes2 or nodes1[1] in nodes2) or (nodes1[1] == nodes2[0]):
                    g_prime.add_edge(e1, e2)

    return g_prime


def create_heterogeneous_graph(individual, curr_step: int = 0, schedule=None, node2vec_args: dict={'embedding_dim': 3, "walk_length": 5, 'context_size': 5,
                                                                                                   'walks_per_node': 10, 'num_negative_samples': 1, 'p': 1, 'q': 1, "sparse": True}):
    """
    Erstellt einen Igraph mit Maschinenknoten (kein disjunktiver Graph). Stattdessen werden Jobs mit den Maschinen
    Knoten verbunden, wenn sie auf den Maschinen gefertigt werden müssen und Jobs in ihrer Operations-Reihenfolge.
    Graph beschrieben in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9826438
    Args:
        individual (list): Liste mit genen.
        curr_step (ind): Bis zu dieser Stelle ist das Individuum Fix (man könnte überlegen hier die alternativen Maschinenkanten
                    weg zu lassen -> bisher aber noch drin). Außerdem wichtig um Candidates für das Modell zu identifizieren.
        schedule (dict): Dict mit ('machine': [(start_time, end_time, job-id, op-id), ...])
    """

    g = ig.Graph(directed=True)

    # Add gene nodes with attributes
    for index, gene in enumerate(individual):
        node_name = f"{gene.job}_{gene.operation}"
        is_inserted = index < curr_step
        start_time, end_time = _find_times(gene.job, gene.operation, schedule)
        durations = [gene.alt_machines[machine] for machine in gene.alt_machines]
        min_duration = min(durations)
        max_duration = max(durations)
        alternatives = len(durations)
        successors = [ind for ind in individual if (ind.job == gene.job) & (ind.operation > gene.operation)]
        work_successors = sum(max(ind.alt_machines.values()) for ind in successors)
        if work_successors is None:
            work_successors = 0
        n_successors = len(successors)
        node_id = g.add_vertex(name=node_name,
                               type=0,
                               inserted=is_inserted,
                               start_time=start_time,
                               end_time=end_time,
                               job=gene.job,
                               operation=gene.operation,
                               min_duration=min_duration,
                               max_duration=max_duration,
                               alternatives=alternatives,
                               successors=n_successors,
                               successor_work=work_successors).index

    machine_nodes = {}
    # Add machine nodes and edges to machines
    for i, gene in enumerate(individual):
        if gene.machine not in machine_nodes:
            if schedule is not None and gene.machine in schedule:
                end_time = schedule[gene.machine][-1][1]
            else:
                end_time = 0
            node_id = g.add_vertex(name=gene.machine,
                                   type=1,
                                   inserted=False,
                                   end_time=end_time,                                       
                                   job=None,
                                   operation=None,
                                   min_duration=-1,
                                   max_duration=-1,
                                   alternatives=-1,
                                   successors=-1,
                                   successor_work=-1).index
            machine_nodes[gene.machine] = node_id

        # Check if the operation can be scheduled next on the target machine
        for alt_machine, duration in gene.alt_machines.items():
            if alt_machine not in machine_nodes:
                node_id = g.add_vertex(name=alt_machine,
                                       type=1,
                                       inserted=False,
                                       start_time=end_time,
                                       end_time=-1,
                                       job=None,
                                       operation=None,
                                       min_duration=-1,
                                       max_duration=-1,
                                       alternatives=-1,
                                       successors=-1,
                                       successor_work=-1).index
                machine_nodes[alt_machine] = node_id

            # Determine if this is a candidate edge and calculate earliest start
            is_candidate = _check_is_candidate(g, gene, alt_machine, individual, curr_step)
            earliest_start = _calculate_earliest_start(g, gene, alt_machine, machine_nodes, individual)
            
            # Check if edge has been chosen or not
            if g.vs()[i]['inserted'] and gene.machine == alt_machine:
                is_chosen = True
                is_discarded = False
            else:
                is_chosen = False
                if i < curr_step:
                    is_discarded = True
                else:
                    is_discarded = False
            # Add Edge
            source_index = individual.index(gene)
            source_node = g.vs()[source_index]
            edge_id = g.add_edge(source_index,
                                 machine_nodes[alt_machine],
                                 type=0,
                                 duration=duration,
                                 is_candidate=is_candidate,
                                 earliest_start=earliest_start,
                                 is_chosen=is_chosen,
                                 is_discarded=is_discarded,
                                 slack_time_added=0,
                                 successors=source_node['successors'],
                                 successors_work=source_node['successor_work'],
                                 open_work_target=0
                                 ).index

    # Add edges for sequential operations
    for i, gene in enumerate(individual):
        next_operations = [ind for ind in individual if (ind.job == gene.job) and (ind.operation > gene.operation)]
        if next_operations:
            next_operation = next_operations[0]
            next_index = individual.index(next_operation)
            
            edge_id = g.add_edge(individual.index(gene),
                                 next_index,
                                 type=1,
                                 is_candidate=False,
                                 duration=-1,
                                 earliest_start=-1,
                                 is_chosen=-1,
                                 is_discarded=-1,
                                 slack_time_added=-1,
                                 successors=-1,
                                 successors_work=-1,
                                 open_work_target=-1
                                 ).index

    g.simplify(multiple=True, loops=True, combine_edges='first')
    

    # Embed Edge Graph in Node to Vec to get edge Embeddings - very slow
    if False:
        edge_graph = create_edge_graph(g, attributes = [])
        nx_graph = edge_graph.to_networkx()
        node2vec = Node2Vec(nx_graph, dimensions=node2vec_args['embedding_dim'], walk_length=node2vec_args["walk_length"], num_walks=200, workers=2, quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = [model.wv[str(i)] for i in range(nx_graph.number_of_nodes())]
        for edge_idx, edge in enumerate(g.es()):
            for dim_idx in range(node2vec_args['embedding_dim']):
                edge[f'n2v_dim_{dim_idx}'] = embeddings[edge_idx][dim_idx]

    return g


def _find_earliest_slot(machine_schedule, duration, min_start_time=0):
    """
    Find the earliest time slot in the schedule that can accommodate the operation.

    Args:
        machine_schedule (list): The schedule for a specific machine.
        duration (int): The duration of the operation.
        min_start_time (int): The minimum start time for the operation.
    
    Returns:
        int, int: The earliest start time for the operation, slack time change
    """
    # Noch kein Schedule
    if len(machine_schedule) == 0:
        # If delayed Start adds slack time
        return max([min_start_time, 0]), max([min_start_time, 0])
    
    # Suche nach Lücke im Schedule
    for idx, (start, end, job, op) in machine_schedule[:-1]:
        possible_start = max([end, min_start_time])
        if machine_schedule[idx + 1][0] - possible_start > duration:
            # Takes away slack time
            return possible_start, (machine_schedule[idx + 1][0] - end) - duration
        
    # Keine Lücke gefunden. Wenn Versatz zum letzten Ende zusätzlicher Slack
    start_time = max((min_start_time, machine_schedule[-1][1]))
    return start_time, start_time - machine_schedule[-1][1]


def _find_times(job, operation, schedule):
    """
    Find the start and end times for a specific job and operation from the schedule.

    Args:
        job (int): The job ID.
        operation (int): The operation ID.
        schedule (dict): The current schedule, a dictionary mapping machines to their scheduled operations.

    Returns:
        tuple: A tuple containing the start and end times for the specified job and operation.
    """
    for machine_schedule in schedule.values():
        for start_time, end_time, job_id, op_id in machine_schedule:
            if job_id == job and op_id == operation:
                return start_time, end_time
    return None, None  # Return None if the job and operation are not found in the schedule


def update_graph_after_planning_step(graph, individual, step, schedule):
    """
    Update graph nodes and edges based on the current step and schedule.

    Args:
        g (igraph.Graph): The graph to be updated.
        individual (list): The current state of the individual.
        step (int): The current step in the algorithm.
        schedule (dict): The current schedule.
    """
    job, operation, machine = individual[step].job, individual[step].operation, individual[step].machine
    # Find node in graph
    node_select = [vertex for vertex in graph.vs() if (vertex['job'] == job) & (vertex['operation'] == operation)]
    if len(node_select) == 1:
        node = node_select[0]
    else:
        raise RuntimeError(f'Multiple or No Genes fit the description Job {job}, OP {operation}')
    
    # Change inserted attribute
    node['inserted'] = True

    start_time, end_time = _find_times(job, operation, schedule)
    node['start_time'] = start_time
    node['end_time'] = end_time

    # Update the earliest possible start time for each edge of type 0
    for edge in graph.es.select(type_eq=0):
        source_node = graph.vs[edge.source]
        target_node = graph.vs[edge.target]

        predecessor = (source_node['job'], source_node['operation'] - 1) if source_node['operation'] > 0 else None
        if predecessor:
            predecessor_start_time, predecessor_end_time = _find_times(predecessor[0], predecessor[1], schedule)
            # Find the earliest empty slot in the schedule that can accommodate the operation
            machine_schedule = schedule.get(source_node['name'].split('_')[1], [])
            if predecessor_end_time is None:
                min_start_time = 0
            else:
                min_start_time = predecessor_end_time
            earliest_start, machine_predecessor_end = _find_earliest_slot(machine_schedule, edge['duration'], min_start_time)
            edge['earliest_start'] = earliest_start
            edge['slack_time_added'] = earliest_start - machine_predecessor_end
    
    # Update machine open work
    machine_worktimes = {}
    for edge in graph.es.select(type_eq=0):
        machine = graph.vs()[edge.target]['name']
        if machine in machine_worktimes.keys():
            edge['open_work_target'] = machine_worktimes[machine]
        else:
            worktimes = [gene.alt_machines[machine] for gene in individual[step:] if machine in gene.alt_machines.keys()]
            machine_worktimes[machine] = sum(worktimes) if len(worktimes) > 0 else 0
            edge['open_work_target'] = machine_worktimes[machine]

    # Update operation-machine edges chosen, candidate
    edges = [edge for edge in graph.es() if (edge['type'] == 0) & (edge.source == node.index)]
    for edge in edges:
        if graph.vs()[edge.target]['name'] == machine:
            edge['is_chosen'] = True
            edge['is_discarded'] = False
        else:
            edge['is_chosen'] = False
            edge['is_discarded'] = True
        edge['is_candidate'] = False


def pad_array(array, padding_size, padding_value, dim=0) -> np.ndarray:
    """
    Padding arrays to a fixed size 

    Args:
        padding_size: Size of the dimension after padding is applied
        padding_value: Value to be inserted
        dim: Dimension for padding
    returns:
        np.ndarray
    """
    padding_size = padding_size - array.shape[dim]
    padding_shape = list(array.shape)
    padding_shape[dim] = padding_size
    padding = np.full(padding_shape, padding_value, dtype=np.float32)
    padded_array = np.concatenate((array, padding), axis=dim)
    return padded_array


def plot_disjunctive_graph(g, target) -> None:
    vertex_colors = ["white" for v in g.vs()]
    edge_colors = []
    for edge in g.es():
        if edge['type'] == 1:
            edge_colors.append('black')
        else:
            edge_colors.append('grey')

    # Visual style
    visual_style = {
        "vertex_size": 25,
        "vertex_color": vertex_colors,
        "vertex_label": g.vs["job_op"],
        "edge_color": edge_colors,
        "bbox": (600, 600),
        "margin": 100,
        "vertex_label_size": 12
    }
    ig.plot(g, target=target, **visual_style)


def order_topological(ind: list) -> list:
    g = build_disjunctive_graph(ind)
    try:
        order = g.topological_sorting()
        return order
    except:
        return None


def graph_to_pytorch_geometric(graph: ig.Graph, node_attributes: list, edge_attributes: list, max_nodes=None, max_edges=None):
    # Node attributes
    node_attributes = np.array([[graph.vs[node][attribute] for attribute in node_attributes] for node in range(graph.vcount())])

    # No Padding
    if max_nodes is None:
        max_nodes = graph.vcount()

    if max_edges is None:
        max_edges = graph.ecount()

    # Pad node array to max_size with -inf
    if graph.vcount() < max_nodes:
        node_attributes = pad_array(node_attributes, max_nodes, float("-inf"))

    # Edge indexes
    edge_index = np.array([edge.tuple for edge in graph.es], dtype=np.float32)

    # Handle max_edges for edge_index
    if max_edges is None:
        max_edges = graph.ecount()

    if graph.ecount() < max_edges:
        edge_index = pad_array(edge_index, max_edges, float("-inf"))
    edge_index = edge_index.transpose()

    # Edge attributes
    edge_attributes = np.array([[graph.es[edge][attribute] for attribute in edge_attributes] for edge in range(graph.ecount())])
    # Handle max_edges for edge_attributes
    if graph.ecount() < max_edges:
        edge_attributes = pad_array(edge_attributes, max_edges, float("-inf"))

    return node_attributes, edge_index, edge_attributes

