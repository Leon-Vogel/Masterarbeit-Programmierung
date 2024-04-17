import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np


def _find_critical_path(G):
    """
    Returns all node indices on critical path. Expects G's vertices to have attributes start_time, end_time.
    """
    operation_ends = [v['end_time'] for v in G.vs()]
    last_ops = np.flatnonzero(operation_ends == np.max(operation_ends)) # In case for multiple nodes with max end time
    critical_operations = []

    def find_critical_predecessor(node):
        predecessor_ops = G.predecessors(node)
        for pre in predecessor_ops:
            if G.vs()['end_time'][pre] == G.vs()['start_time'][node]:
                critical_operations.append(pre)
                find_critical_predecessor(pre)

    for end_node in last_ops:
        find_critical_predecessor(end_node)
    return list(set(critical_operations))
                

def plot_graph(G):
    pos = nx.circular_layout(G)
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)
    edge_labels = nx.get_edge_attributes(G, 'edge_type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    edge_labels = nx.get_edge_attributes(G, 'time_dist')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.4)

    # edge_labels = nx.get_edge_attributes(G, 'next_example_weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.6)
    
    nx.draw_networkx(G, pos, with_labels=True, node_size=500, font_size=16, font_weight='bold')
    plt.axis('off')
    plt.show()

def generate_random_solution(instance, n_machines):
    """
    Generiert random Lösung für Hurink instance
    """
    schedule = []
    jobs = [j['job'] for j in instance]
    current_machine_times = np.zeros(n_machines)
    current_job_times = np.zeros(len(jobs))
    while len(jobs) > 0:
        random_job = np.random.choice(jobs)
        jobdata = next(filter(lambda j: j['job'] == random_job, instance))
        ops = jobdata['ops']
        next_op = ops.pop(0)
        machine_data = np.random.choice(next_op['machines'])
        machine = machine_data['machine']
        machine_idx = int(machine[1:]) - 1
        job_idx = int(random_job[1:]) - 1
        duration = machine_data['dur']
        machine_available_at = current_machine_times[machine_idx]
        job_available_at = current_job_times[job_idx]
        start_time = max((machine_available_at, job_available_at))
        end_time = start_time + duration
        entry = {
            "start_time": start_time,
            "end_time": end_time,
            "resource": int(machine[1:]),
            "job": random_job,
            "op": f"{random_job[1:]}_{next_op['i']}",
            "machine_choices": next_op['machines']
        }
        current_machine_times[machine_idx] = end_time
        current_job_times[job_idx] = end_time
        if len(ops) == 0:
            jobs.remove(random_job)
        schedule.append(entry)
    return schedule

def schedule_data_to_nxgraph(sched_dat):
    """
    Expects data in this format (also used to create gantt diagrams):

    sched_dat["operations"] = [
        {
            "start_time": 0,
            "end_time: 5,
            "resource": 17,
            "job": 25,
        },
    ]
    """
    if type(sched_dat) is dict:
        assert "operations" in sched_dat, 'the key "operations" must be available.'
        sched_dat = list(sched_dat["operations"])
    if type(sched_dat) is list:
        for o1 in sched_dat:
            assert (
                "start_time" in o1 and "end_time" in o1 and "resource" in o1 and "job" in o1
            ), f"{o1} has missing entries."
    else:
        raise Exception("Gantt data must be list of operations or dictionary with list of operations")

    G = nx.DiGraph()
    for i in range(len(sched_dat)):
        info = [sched_dat[i][attr] for attr in ['start_time', 'end_time']]
        G.add_node(i, info=info, jro=f"J{sched_dat[i]['job']}_R{sched_dat[i]['resource']}_OP{sched_dat[i]['op']}")

    # now generate a graph (DAG) with two edge types (ops_type):
    # edge type 1 (resource operations): operations have resource predecessors and successors. this is depicted with an directed edge
    # edge type 2 (job operation): operations have preceeding and succeeding operations within the job. this is also depicted via edges

    # the edge weight is the difference of succeeding operation start time and preceeding operation end time

    for i, o1 in enumerate(sched_dat):
        other_ops = list(filter(lambda x: sched_dat.index(x) != i, sched_dat))
        resource_ops = list(filter(lambda x: x["resource"] == o1["resource"], other_ops))
        job_ops = list(filter(lambda x: x["job"] == o1["job"], other_ops))

        for ops in [(1, resource_ops), (2, job_ops)]:
            ops_type = ops[0]
            ops = sorted(ops[1], key=lambda x: x["start_time"])

            def get_related_op(get_pre):
                related_ops = (
                    list(filter(lambda x: x["start_time"] < o1["start_time"], ops))
                    if get_pre
                    else list(filter(lambda x: x["start_time"] > o1["start_time"], ops))
                )
                if related_ops:
                    return related_ops[-1] if get_pre else related_ops[0]
                return None

            pre = get_related_op(get_pre=True)
            if pre != None:
                pre_i = sched_dat.index(pre)
                G.add_edge(pre_i, i, time_dist=o1["start_time"] - pre["end_time"], edge_type=ops_type, weight=1)
            
            suc = get_related_op(get_pre=False)
            if suc != None:
                suc_i = sched_dat.index(suc)
                G.add_edge(i, suc_i, time_dist=suc["start_time"] - o1["end_time"], edge_type=ops_type, weight=1)

    # add node attributes:
    for node in G.nodes():
        node_data = nx.get_node_attributes(G, 'info')[node]
        job_predecessor_edges = [e for e in G.edges(data=True) if e[1] == node and len(e) > 2 and e[2]['edge_type'] == 2]
        job_successor_edges = [e for e in G.edges(data=True) if e[0] == node and len(e) > 2 and e[2]['edge_type'] == 2]
        res_predecessor_edges = [e for e in G.edges(data=True) if e[1] == node and len(e) > 2 and e[2]['edge_type'] == 1]
        res_successor_edges = [e for e in G.edges(data=True) if e[0] == node and len(e) > 2 and e[2]['edge_type'] == 1]
        # Wait for job predecessors
        if len(job_predecessor_edges):
            node_data.append(min([e[2]['time_dist'] for e in job_predecessor_edges]))
        else:
            node_data.append(-1)

        # Wait to next Successor Job
        if len(job_successor_edges):
            node_data.append(min([e[2]['time_dist'] for e in job_successor_edges]))
        else:
            node_data.append(-1)

        # Wait to resource Predecessor
        if len(res_predecessor_edges):
            node_data.append(min([e[2]['time_dist'] for e in res_predecessor_edges]))
        else:
            node_data.append(-1)

        # Wait to resource Successor
        if len(res_successor_edges):
            node_data.append(min([e[2]['time_dist'] for e in res_successor_edges]))
        else:
            node_data.append(-1)
        
        nx.set_node_attributes(G, {node: {'info': node_data}})

    # returns the graph G and the edge attributes
    return G, ['time_dist', 'edge_type']

def schedule_data_to_igraph(sched_dat):
    """
    Expects data in this format (also used to create gantt diagrams):

    sched_dat["operations"] = [
        {
            "start_time": 0,
            "end_time: 5,
            "resource": 17,
            "job": 25,
            "machine_choices": [{'machine': m2, 'dur': 93}]
        },
    ]
    """
    if type(sched_dat) is dict:
        assert "operations" in sched_dat, 'the key "operations" must be available.'
        sched_dat = list(sched_dat["operations"])
    if type(sched_dat) is list:
        for o1 in sched_dat:
            assert (
                "start_time" in o1 and "end_time" in o1 and "resource" in o1 and "job" in o1
            ), f"{o1} has missing entries."
    else:
        raise Exception("Gantt data must be list of operations or dictionary with list of operations")

    G = ig.Graph().as_directed()
    for i in range(len(sched_dat)):
        G.add_vertex(str(i), start_time=sched_dat[i]['start_time'],
                    end_time=sched_dat[i]['end_time'],
                    job=sched_dat[i]['job'],
                    resource=sched_dat[i]['resource'],
                    op=sched_dat[i]['op'],
                    jro=f"J{sched_dat[i]['job']}_R{sched_dat[i]['resource']}_OP{sched_dat[i]['op']}",
                    features=[sched_dat[i]['start_time'], sched_dat[i]['end_time'], sched_dat[i].get('machine_choices')])

    # now generate a graph (DAG) with two edge types (ops_type):
    # edge type 1 (resource operations): operations have resource predecessors and successors. this is depicted with an directed edge
    # edge type 2 (job operation): operations have preceeding and succeeding operations within the job. this is also depicted via edges

    # the edge weight is the difference of succeeding operation start time and preceeding operation end time

    for i, o1 in enumerate(sched_dat):
        other_ops = list(filter(lambda x: sched_dat.index(x) != i, sched_dat))
        resource_ops = list(filter(lambda x: x["resource"] == o1["resource"], other_ops))
        job_ops = list(filter(lambda x: x["job"] == o1["job"], other_ops))

        for ops in [(1, resource_ops), (2, job_ops)]:
            ops_type = ops[0]
            ops = sorted(ops[1], key=lambda x: x["start_time"])

            def get_related_op(get_pre):
                related_ops = (
                    list(filter(lambda x: x["start_time"] < o1["start_time"], ops))
                    if get_pre
                    else list(filter(lambda x: x["start_time"] > o1["start_time"], ops))
                )
                if related_ops:
                    return related_ops[-1] if get_pre else related_ops[0]
                return None

            pre = get_related_op(get_pre=True)
            if pre != None:
                pre_i = sched_dat.index(pre)
                pre_machine = pre['resource']
                G.add_edge(pre_i, i, time_dist=o1["start_time"] - pre["end_time"], edge_type=ops_type, weight=1, source_machine=pre_machine, target_machine=ops['resource'])
            
            suc = get_related_op(get_pre=False)
            if suc != None:
                suc_i = sched_dat.index(suc)
                suc_machine = suc['resource']
                G.add_edge(i, suc_i, time_dist=suc["start_time"] - o1["end_time"], edge_type=ops_type, weight=1, source_machine=ops['resource'], target_machine=suc_machine)

    G = G.simplify(multiple=True, loops=True, combine_edges='first')
    critical_path_nodes = _find_critical_path(G)
    # add node attributes:
    for node in G.vs():
        node_data = node['features']
        job_predecessor_edges = [e for e in G.es() if e['edge_type'] == 2 and e.target == int(node['name'])]
        job_successor_edges = [e for e in G.es() if e['edge_type'] == 2 and e.source == int(node['name'])]
        res_predecessor_edges = [e for e in G.es() if e['edge_type'] == 1 and e.target == int(node['name'])]
        res_successor_edges = [e for e in G.es() if e['edge_type'] == 1 and e.source == int(node['name'])]
        # Wait for job predecessors
        if len(job_predecessor_edges):
            node_data.append(min([e['time_dist'] for e in job_predecessor_edges]))
        else:
            node_data.append(-1)

        # Wait to next Successor Job
        if len(job_successor_edges):
            node_data.append(min([e['time_dist'] for e in job_successor_edges]))
        else:
            node_data.append(-1)

        # Wait to resource Predecessor
        if len(res_predecessor_edges):
            node_data.append(min([e['time_dist'] for e in res_predecessor_edges]))
        else:
            node_data.append(-1)

        # Wait to resource Successor
        if len(res_successor_edges):
            node_data.append(min([e['time_dist'] for e in res_successor_edges]))
        else:
            node_data.append(-1)

        # Resource Utilization
        node_start = node['start_time']
        node_duration = node['end_time'] - node_start
        resource_ops = list(filter(lambda x: (x["resource"] == o1["resource"]) & (x['start_time'] < node_start), sched_dat))
        worktimes = [operation["end_time"] - operation['start_time'] for operation in resource_ops]

        utilization = (node['end_time'] - (sum(worktimes) + node_duration)) / node['end_time']
        node_data.append(utilization)
        # Duration Normalized, Machine Alternatives
        if node['features'][2] is not None: # Machine Choices exist in data
            min_duration = min([c['dur'] for c in node['features'][2]])
            efficiency = min_duration / node_duration
            node_data.append(efficiency)
            node_data.append(len(node['features'][2]))
        else:
            node_data.append(None)
            node_data.append(None)

        # Critical Path
        node_data.append(node in critical_path_nodes)
        
        node['features'] = node_data

    # returns the graph G and the edge attributes
    return G, ['time_dist', 'edge_type', 'weight', 'source_machine', 'target_machine']

import torch
def test(graph):
    node_features = []
    for node in graph.nodes():
        out_edges = graph.out_edges(node, data=True)
        if len(out_edges) == 0:
            node_feature = torch.tensor([0, 0], dtype=torch.float)  # Platzhalter, falls es keine ausgehenden Kanten gibt
        else:
            time_dists = [edge_data['time_dist'] for _, _, edge_data in out_edges]
            edge_types = [edge_data['edge_type'] for _, _, edge_data in out_edges]
            node_feature = torch.tensor([sum(time_dists)/len(time_dists), sum(edge_types)/len(edge_types)], dtype=torch.float)
        node_features.append(node_feature)

    x = torch.stack(node_features)

if __name__ == "__main__":
    test = [
        {
            'start_time': 0,
            'end_time': 2,
            'resource': 1,
            'job': 10,
            'op': 1
        },
        {
            'start_time': 0,
            'end_time': 1,
            'resource': 2,
            'job': 20,
            'op': 1
        },
        {
            'start_time': 3,
            'end_time': 4,
            'resource': 1,
            'job': 20,
            'op': 2
        },
        {
            'start_time': 4,
            'end_time': 8,
            'resource': 2,
            'job': 10,
            'op': 2
        },
    ]
    graph, edge_attr = schedule_data_to_igraph(test)
    graph, edge_attr = schedule_data_to_nxgraph(test)
    plot_graph(graph)
    test(graph)