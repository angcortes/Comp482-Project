import heapq
import random
import time
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import ListedColormap

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

# used in A* to get distance to goal from current node
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

# Animation functions
def setup_live_grid(size, walls, title):
  plt.ion()
  fig, ax = plt.subplots(figsize=(6, 6))
  
  # Initialize Grid: 0=Empty, 1=Wall
  grid_map = [[0] * size for _ in range(size)]
  for (wx, wy) in walls:
      grid_map[wx][wy] = 1
  
  # Setup Colors: 0=White, 1=Black, 2=Blue (Visited), 3=Green (Path/Start/End)
  cmap = ListedColormap(['white', 'black', 'dodgerblue', 'lime'])
  
  # Create the image object (we will update this object's data later)
  img = ax.imshow(grid_map, cmap=cmap, origin='upper', vmin=0, vmax=3)
  
  ax.set_title(title)
  ax.axis('off')
  plt.show()
  plt.pause(0.1) # Give the window a moment to appear
  
  return img, grid_map

# Updates the plot with the visited node. Frequency controls speed
def update_live_grid(img, grid_map, current_node, nodes_explored, frequency=10):
    # Only update every Nth node to prevent lag
    if nodes_explored % frequency == 0:
        # Set pixel to 'Visited' color (2)
        grid_map[current_node[0]][current_node[1]] = 2
        
        # Push data to the plot
        img.set_data(grid_map)
        
        # Tiny pause to let the OS redraw the window
        # 0.001 is fast, increase to 0.01 to slow it down
        plt.pause(0.1)

## ALGOS
def dijkstra(graph, start_node, goal_node, size, walls, visualize=False):

  # setup visulation
  if visualize:
    img, grid_map = setup_live_grid(size, walls, "Dijkstra Visual")

  prio_queue = [(0, start_node)]
  visited_cost = {start_node: 0}
  came_from = {}
  nodes_explored = 0

  while prio_queue:
    cost, current_node = heapq.heappop(prio_queue)
    nodes_explored += 1

    # update visualization
    if visualize:
            # frequency=20 means draw every 20 steps. Lower = smoother but slower.
            update_live_grid(img, grid_map, current_node, nodes_explored, frequency=1)

    if current_node == goal_node:
      # turns of interactive mode
      if visualize:
        plt.ioff()
        plt.show()

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

  if visualize:
        print("Path not found! Keeping window open.")
        plt.ioff()
        plt.show()
  return float('inf'), nodes_explored, []

def a_star(graph, start, goal, size, walls, visualize=False):

  # visualization setup
  if visualize:
        img, grid_map = setup_live_grid(size, walls, "A* Visual")

  prio_queue = [(0,0, start)]
  g_score = {start: 0}
  came_from = {}
  nodes_explored = 0

  while prio_queue:
    # f_score only matters for the order it will be explored
     _,_, current_node = heapq.heappop(prio_queue)
     nodes_explored += 1

     if visualize:
        update_live_grid(img, grid_map, current_node, nodes_explored, frequency=1)     
    

     if current_node == goal:
      if visualize:
        plt.ioff()
        plt.show()
      return g_score[current_node], nodes_explored, reconstruct_path(came_from, start, goal)
     
     for neighbor_node, weight in graph.get(current_node, {}).items():
       
       tentative_g_score = g_score[current_node] + weight

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
         heapq.heappush(prio_queue, (f_score, h, neighbor_node))
  
  if visualize:
        print("Path not found! Keeping window open.")
        plt.ioff()
        plt.show()

  return float('inf'), nodes_explored, []

def bidirectional_dijkstra(graph, start_node, goal_node, size, walls, visualize=False):

  # visualization setup
  if visualize:
        img, grid_map = setup_live_grid(size, walls, "Bidirectional Dijkstra Visual")

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

    # visual update
    if visualize:
             # Uses shared_state counter for frequency check
             update_live_grid(img, grid_map, current_node, shared_state['num_nodes_explored'], frequency=1)

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
      
      # check if the smallest path in the prio queue can beat the current smallest connected path
      if smallest_in_f_prio_que + smallest_in_b_prio_que >= shared_converge_state['best_path_cost']:
        if visualize:
           plt.ioff()
           plt.show()
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

  if visualize:
        print("Path not found! Keeping window open.")
        plt.ioff()
        plt.show()
  return float('inf'), shared_converge_state['num_nodes_explored'], []

def bidirectional_a_star(graph, start, goal, size, walls, visualize=False):
    
  # setup animated visualization
  if visualize:
      img, grid_map = setup_live_grid(size, walls, "Bidirectional A* visual")
  else:
      img, grid_map = None, None

  # Heuristic Functions
  # Forward search aims for Goal
  def heur_fwd(n): 
    return calc_heuristic(n, goal)
  # Backward search aims for Start
  def heur_bwd(n): 
    return calc_heuristic(n, start)

  start_h = heur_fwd(start)
  goal_h = heur_bwd(goal)

   # Priority Queues: Stores (f_score, h_score, node)
  # We include h_score for tie-breaking
  prio_que_forw = [(start_h, start_h, start)]
  prio_que_back = [(goal_h, goal_h, goal)]
  
  # G-Scores (The real distance traveled)
  g_forw = {start: 0}
  g_back = {goal: 0}

  # Parents for path reconstruction
  parent_forw = {start: None}
  parent_back = {goal: None}

  shared_state = {
      'num_nodes_explored': 0,
      'meet_node': None,
      'best_path_cost': float('inf')
  }

  def expand(prio_queue, g_scores, parent, other_g_scores, get_heuristic):
      # Pop the node with lowest F-Score
      # (f, h, node)
      # only the current_node matters once popped
      _, _, current_node = heapq.heappop(prio_queue)
      
      shared_state['num_nodes_explored'] += 1

      # Visualization
      if visualize:
          update_live_grid(img, grid_map, current_node, shared_state['num_nodes_explored'], frequency=1)

      # CHECK FOR CONNECTION
      if current_node in other_g_scores:
          total = g_scores[current_node] + other_g_scores[current_node]
          if total < shared_state['best_path_cost']:
              shared_state['best_path_cost'] = total
              shared_state['meet_node'] = current_node
      
      # If this path is already worse than the best known complete path, stop.
      # We compare g-score (real cost), not f-score here for safety
      if g_scores[current_node] >= shared_state['best_path_cost']:
          return

      # itterate through current nodes connected neighbors
      for neighbor, weight in graph.get(current_node, {}).items():
          tentative_g = g_scores[current_node] + weight
          
          if tentative_g < g_scores.get(neighbor, float('inf')):
              g_scores[neighbor] = tentative_g
              parent[neighbor] = current_node
              
              # Calculate F-Score = G + H
              h_val = get_heuristic(neighbor)
              f_val = tentative_g + h_val
              
              # Push (f, h, node) for tie-breaking
              heapq.heappush(prio_queue, (f_val, h_val, neighbor))

  while prio_que_forw and prio_que_back:
      # Expand Forward (Using Forward Heuristic)
      expand(prio_que_forw, g_forw, parent_forw, g_back, heur_fwd)

      # Expand Backward (Using Backward Heuristic)
      expand(prio_que_back, g_back, parent_back, g_forw, heur_bwd)

      # STOPPING CONDITION
      # If the smallest F-value in either queue is >= best_path_cost, 
      # we can't possibly find a better path.
      if shared_state['best_path_cost'] != float('inf'):
          min_f_fwd = prio_que_forw[0][0] if prio_que_forw else float('inf')
          min_f_bwd = prio_que_back[0][0] if prio_que_back else float('inf')
          
          # We take the MAX of the mins because if EITHER side's minimum potential 
          # is worse than our current best path, that side can't contribute a better solution. 
          if min(min_f_fwd, min_f_bwd) >= shared_state['best_path_cost']:
              if visualize: plt.ioff(); plt.show()
              return (shared_state['best_path_cost'], 
                      shared_state['num_nodes_explored'],
                      build_bi_path(parent_forw, parent_back, shared_state['meet_node'], start, goal))

  if visualize:
        print("Path not found! Keeping window open.")
        plt.ioff()
        plt.show()
  return float('inf'), shared_state['num_nodes_explored'], []

##CREATING THE GRID

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

def save_image(size, walls, path, title, filename):
    grid_map = [[0] * size for _ in range(size)]
    
    for (wx, wy) in walls:
        grid_map[wx][wy] = 1 
    
    if path:
        for (px, py) in path:
            grid_map[px][py] = 2 
            
    grid_map[0][0] = 3 
    grid_map[size-1][size-1] = 3 

    plt.figure(figsize=(5, 5))
    
   
    cmap = ListedColormap(['white', 'black', 'dodgerblue', 'lime'])
    
    plt.imshow(grid_map, cmap=cmap, origin='upper')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    print(f"   -> Saved image: {filename}")
    plt.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Home Assignment2-COMP482")
    parser.add_argument("-s", "--size", type=int, default=20, help="size of the axes on the grid(they are symmetrical)")
    parser.add_argument("-v", "--visual", action="store_true", help="allow visualization of algos on the graph")
    parser.add_argument("-o","--obstacle", type=float, default = .15, help="decimal percantage chance of obstacle appearing on grid")
    args = parser.parse_args()


    SIZE = args.size
    OBSTACLE_PROBABILITY = args.obstacle
    DO_WE_WANT_VISUAL_RUN = args.visual

    print(f"\n Racing on a {SIZE}x{SIZE} Grid Maze")
    grid_graph, start, goal, walls = create_grid(SIZE, obstacle_prob= OBSTACLE_PROBABILITY)
    
    print("-" * 65)
    print(f"{'ALGORITHM':<25} | {'TIME (sec)':<10} | {'VISITED':<10} | {'COST'}")
    print("-" * 65)

    #live demo
    if DO_WE_WANT_VISUAL_RUN:
      bidirectional_a_star(grid_graph, start, goal, SIZE, walls, visualize=True)
      a_star(grid_graph, start, goal, SIZE, walls, visualize=True)
      bidirectional_dijkstra(grid_graph, start, goal, SIZE, walls, visualize=True)

    # Dijkstra
    t0 = time.time()
    cost, visited, path = dijkstra(grid_graph, start, goal, SIZE, walls)
    print(f"{'Dijkstra':<25} | {time.time()-t0:.5f}     | {visited:<10} | {cost}")
    save_image(SIZE, walls, path, f"Dijkstra (Visited: {visited})", "dijkstra.png")

    # A*
    t0 = time.time()
    cost, visited, path = a_star(grid_graph, start, goal,SIZE, walls)
    print(f"{'A* (Manhattan)':<25} | {time.time()-t0:.5f}     | {visited:<10} | {cost}")
    save_image(SIZE, walls, path, f"A-Star (Visited: {visited})", "astar.png")

    # Bidirectional Dijkstra
    t0 = time.time()
    cost, visited, path = bidirectional_dijkstra(grid_graph, start, goal, SIZE, walls)
    print(f"{'Bidirectional':<25} | {time.time()-t0:.5f}     | {visited:<10} | {cost}")
    save_image(SIZE, walls, path, f"Bidirectional (Visited: {visited})", "bidirectional_dijkstra.png")

    # Bidirectional A*
    t0 = time.time()
    cost, visited, path = bidirectional_a_star(grid_graph, start, goal, SIZE, walls)
    print(f"{'Bidirectional A*':<25} | {time.time()-t0:.5f}     | {visited:<10} | {cost}")
    save_image(SIZE, walls, path, f"Bidirectional A* (Visited: {visited})", "bidirectional_a_star.png")


