import heapq


def reconstruct_path(came_from, start, current_node):

  path = []
  while current_node in came_from:
    path.append(current_node)
    current = came_from[current]

  path.append(start)
  path.reverse()
  return path

def dijkstra(graph, start_node, goal_node):

  prio_queue = [(0, start_node)]
  visited_cost = {start_node: 0}
  came_from = {}
  nodes_explored = 0

  while prio_queue:
    cost, current_node = heapq.heappop(prio_queue)
    nodes_explored += 1

    if current_node == goal_node:
      ## reconstruct path
      return visited_cost[current_node], nodes_explored, reconstruct_path(came_from, start_node, goal_node)

    if cost > visited_cost[current_node]:
      continue

    for neighbor_node, weight in graph.get(current_node, {}).items():
      new_cost = cost + weight
      
      # if visited isn't within visited_cost return default infinity value
      # dynamically fills unvisted nodes to reduce iterations
      if new_cost < visited_cost.get(neighbor_node, float('inf')):
        visited_cost[neighbor_node] = new_cost
        came_from[neighbor_node] = current_node
        heapq.heappush(prio_queue, (new_cost, neighbor_node))


