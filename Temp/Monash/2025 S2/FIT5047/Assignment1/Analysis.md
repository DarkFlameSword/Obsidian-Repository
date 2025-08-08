# Pacman AI Search Project Documentation

## Project Overview

This is a Python-based implementation of the classic Pacman game with AI search algorithms. The project is structured to support different search problems and solvers for educational purposes, likely for an AI/Machine Learning course (FIT5047).

## Project Structure

### Core Game Files
- `pacman.py` - Main game engine and `GameState` class
- `game.py` - Core game logic, Actions, Agent, and Directions classes
- `layout.py` - Maze layout management
- `util.py` - Utility functions and data structures

### Display and Interface
- `graphicsDisplay.py` - Graphical game display
- `graphicsUtils.py` - Graphics utility functions  
- `textDisplay.py` - Text-based game display

### Agent Implementations
Located in `agents/` directory:
- `pacmanAgents.py` - Pacman agent implementations
- `searchAgents.py` - Search-based agents
- `ghostAgents.py` - Ghost AI agents
- `greedyAgent.py` - Greedy search agent
- `keyboardAgents.py` - Human-controllable agents
- `q2Agent.py` - Question 2 specific agent

### Problem Definitions
Located in `problems/` directory:
- `q1a_problem.py` - Basic pathfinding problem
- `q1b_problem.py` - Corners problem (visiting all corners)
- `q1c_problem.py` - Food search problem

### Search Solvers
Located in `solvers/` directory:
- `q1a_solver.py` - Solver for basic pathfinding
- `q1b_solver.py` - Solver for corners problem
- `q1c_solver.py` - Solver for food search

### Maze Layouts
Located in `layouts/` directory with categorized maze files:
- **Q1a layouts**: Basic pathfinding mazes (`q1a_*.lay`)
- **Q1b layouts**: Corner-visiting mazes (`q1b_*.lay`) 
- **Q1c layouts**: Food collection mazes (`q1c_*.lay`)
- **Q2 layouts**: Game-playing scenarios (`q2_*.lay`)

### Logging and Evaluation
- `logs/search_logger.py` - Search algorithm logging
- `evaluator.py` - Performance evaluation
- `testParser.py` - Test case parsing

## Code Analysis

### Q1a Problem Class (`problems/q1a_problem.py`)

The `q1a_problem` class represents a basic search problem for pathfinding in Pacman:

```python
class q1a_problem:
    def __init__(self, gameState: GameState):
        self.startingGameState: GameState = gameState
```

**Key Features:**
- **State Space**: (x,y) positions on the Pacman board
- **Constructor**: Stores the initial game state for reference
- **Logging**: Uses `@log_function` decorator for method tracking
- **Incomplete Implementation**: Contains placeholder methods that need implementation:
  - `getStartState()` - Should return the starting position
  - `isGoalState(state)` - Should check if a state is the goal
  - `getSuccessors(state)` - Should return valid moves with costs

**Method Structure:**
- All search methods raise `util.raiseNotDefined()`, indicating this is a template for students to complete
- Uses type hints for better code documentation
- Follows standard search problem interface with start state, goal test, and successor function

### Project Architecture

The project follows a **separation of concerns** design:

1. **Game Engine Layer**: Core game mechanics (`pacman.py`, `game.py`)
2. **Problem Definition Layer**: Abstract search problems (`problems/`)
3. **Solver Layer**: Search algorithm implementations (`solvers/`)
4. **Agent Layer**: Decision-making entities (`agents/`)
5. **Presentation Layer**: Display and user interface (`graphicsDisplay.py`, `textDisplay.py`)

### Educational Structure

The project is designed for incremental learning:
- **Q1a**: Basic pathfinding (find shortest path to goal)
- **Q1b**: Multi-goal problems (visit all corners)
- **Q1c**: Complex search spaces (collect all food)
- **Q2**: Game tree search and adversarial agents

This structure allows students to progress from simple search problems to complex multi-agent scenarios while reusing the same game framework.