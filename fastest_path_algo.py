import heapq
from typing import Tuple

# Types
Node = Tuple[int,int]


def reconstruct_path(came_from, start, current_node):

  path = []
  while current_node in came_from:
    path.append(current_node)
    current = came_from[current]

  path.append(start)
  path.reverse()
  return path


def calc_heuristic(current_node, goal_node):
  curr_node_x = current_node[0]
  curr_node_y = current_node[1]
  goal_node_x = goal_node[0]
  goal_node_y = goal_node[1]

  # uses manhattan method to get distance
  distance_goal = abs(curr_node_x - goal_node_x) + abs(curr_node_y - goal_node_y)
  return distance_goal


## ALGOS
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

        # keeping track of path to parrent node
        came_from[neighbor_node] = current_node
        heapq.heappush(prio_queue, (new_cost, neighbor_node))

def a_star(graph, start:Node, goal:Node):

  prio_queue = [(0, start)]
  g_score = {start: 0}
  came_from = {}
  nodes_explored = 0

  while prio_queue:
    # f_score only matters for the order it will be explored
     _, current_node = heapq.heappop(prio_queue)
     nodes_explored += 1

     if current_node == goal:
       return g_score[current_node], nodes_explored, reconstruct_path(came_from, start, goal)
     
     for neighbor_node, weight in graph.get(current_node, {}).items():
       
       tentative_g_score = g_score[current_node] + weight
       nodes_explored += 1

       if tentative_g_score < g_score.get(neighbor_node, float('inf')):
         g_score[neighbor_node] = tentative_g_score

         # keeping track of path to parrent node
         came_from[neighbor_node] = current_node

         # f = g + h
         # important difference
         h = calc_heuristic(neighbor_node, goal)
         f_score = tentative_g_score + h

         ## adds the f score to the priority queue
         ## using f score is what makes it prioritize nodes that are closer to the goal
         heapq.heappush(prio_queue, (f_score, neighbor_node))

def bidirectional_dijkstra(graph, start_node, goal_node):

  prio_que_forw = [(0, start_node)]
  prio_que_back = [(0, goal_node)]
  
  vis_cost_forw = {start_node:0}
  vis_cost_back = {goal_node:0}

  shared_convergence_state = {
    'num_nodes_explored': 0,
    'meet_node': None,
    'best_cost': float('inf')
  }
  

