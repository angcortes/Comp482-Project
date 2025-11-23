import heapq
import random
from typing import Tuple

# Types
Node = Tuple[int,int]

# HELPER FUNCTIONS
def reconstruct_path(came_from, start, current_node, isForward=True):

  path = []
  while current_node in came_from:
    path.append(current_node)
    current_node = came_from[current_node]

  path.append(start)

  if isForward:
    path.reverse()

  return path


def calc_heuristic(current_node, goal_node):
  curr_node_x = current_node[0]
  curr_node_y = current_node[1]
  goal_node_x = goal_node[0]
  goal_node_y = goal_node[1]

  # uses manhattan method to get distance
  distance_to_goal = abs(curr_node_x - goal_node_x) + abs(curr_node_y - goal_node_y)
  return distance_to_goal

# builds a connected path from the two paths
def build_bi_path(parent_forward_dic, parent_back_dic, meet_node, start_node, goal_node):
  if meet_node is None:
    return []
  
  forward_path = reconstruct_path(parent_forward_dic, start_node, meet_node)
  backward_path = reconstruct_path(parent_back_dic, goal_node, meet_node, isForward=False)

  # combine the path
  # remove the repetition in the backward path
  full_path = forward_path + backward_path[1:]

  return full_path
  

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
  
  vis_forw = {start_node:0}
  vis_back = {goal_node:0}

  parent_forw = {}
  parent_back = {}

  shared_converge_state = {
    'num_nodes_explored': 0,
    'meet_node': None,
    'best_path_cost': float('inf')
  }

  def expand(prio_queue, parent, visited, other_search_visited, shared_state):

    cost, current_node = heapq.heappop(prio_queue)
    shared_state['num_nodes_explored'] += 1

    # use other searches visited to check if a connection has been found
    if current_node in other_search_visited:
      total = cost + other_search_visited[current_node]

      if total < shared_state['best_path_cost']:
        shared_state['best_path_cost'] = total
        shared_state['meet_node'] = current_node

    if cost >= shared_state['best_path_cost']:
        return
      
    for neighbor_node, weight in graph.get(current_node, {}).items():
      new_cost = cost + weight

      if new_cost < visited.get(neighbor_node, float('inf')):
        visited[neighbor_node] = new_cost
        parent[neighbor_node] = current_node

        heapq.heappush(prio_queue, (new_cost, neighbor_node))

  # loops while neither prio_queue is empty
  while prio_que_forw and prio_que_back:

    # Expands from starting node
    expand(prio_que_forw,parent_forw, vis_forw, vis_back, shared_converge_state)

    # Expands from goal node
    expand(prio_que_back, parent_back, vis_back, vis_forw, shared_converge_state)

    # stopping condition
    # start checking stopping condition once a connection has been formed
    if shared_converge_state['best_path_cost'] != float('inf'):
      smallest_in_f_prio_que = None
      smallest_in_b_prio_que = None

      # grab the smallest element in the prio queues
      if prio_que_forw:
        smallest_in_f_prio_que = prio_que_forw[0][0]
      else:
        smallest_in_f_prio_que = float('inf')
      if prio_que_back:
        smallest_in_b_prio_que = prio_que_back[0][0]
      else:
        smallest_in_b_prio_que = float('inf')
      
      if smallest_in_f_prio_que + smallest_in_b_prio_que >= shared_converge_state['best_path_cost']:
        return (shared_converge_state['best_path_cost'], 
                shared_converge_state['num_nodes_explored'],
                build_bi_path(
                  parent_forw,
                  parent_back,
                  shared_converge_state['meet_node'],
                  start_node,
                  goal_node
                )
                )


## creates 2 dimensional graph with walls(holes)
def create_grid(size, obstacle_prob=0.3):

  graph = {}
  nodes = set()
  walls = set()

  start = (0,0)
  goal = (size -1, size-1)

  for x in range(size):
    for y in range(size):
      current = (x,y)

      ## make sure start and goal are not walls
      if current == start or current == goal:
        nodes.add(current)
        continue
      
      # randomly add disconnected nodes to act as walls
      if random.random() > obstacle_prob:
        nodes.add(current)
      else:
        walls.add(current)
  
  for x,y in nodes:
    graph[(x,y)] = {}

    # check nodes in all 4 directions of current node
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      neighbor = (x + dx, y + dy)

      # check if neighbor is a node
      if neighbor in nodes:
        graph[(x, y)][neighbor] = 1
  
  return graph, start, goal, walls
        


