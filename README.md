# Table_top_robot_co

A MuJoCo-based simulation of a **4-wheel toy robot** that moves freely on a 5×5 tabletop — it can move forward, rotate left or right, and report its position, but it’s always protected from falling off the table.

---

## Features

### Command Interface

* **PLACE X,Y,FACING** – Places the robot at grid coordinates (X,Y) facing NORTH, SOUTH, EAST, or WEST.
* **MOVE** – Moves the robot forward by one grid unit.
* **LEFT / RIGHT** – Rotates the robot 90° left or right.
* **REPORT** – Prints the robot's current position and orientation.
* **QUIT** – Exits the simulation.

### Safety

* Prevents the robot from falling off the tabletop.
* Inhibits move and rotation commands that would cause unsafe motion.
* Automatically halts rotation when too close to edges.
* Debounce logic ensures smooth switching between move and rotate states.

### Testing

* **Unit tests** for all helper modules and the controller.
* **Integration tests** running the full simulation with fake user input.

---

## Setup & Installation

### Requirements

* Python 3.12+
* [MuJoCo](https://mujoco.org/)

### Installation

Clone the repository:
```bash
https://github.com/AbhijithSree1/Table_top_robot_co.git
```

From the repository root:

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Simulation

From the repository root:

Linux:

```bash
python3 -m src.ToyRobotControl_main
```

Windows (PowerShell):

```bash
python -m src.ToyRobotControl_main
```

Once the MuJoCo viewer opens, type commands into the terminal:

```
PLACE 1,1,NORTH
MOVE
RIGHT
REPORT
QUIT
```

**Example Output:**

```
robot at x=1 y=2 facing EAST
```

---

## Running Tests

Run all tests (unit + integration):

From the repository root:

Linux:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v
```

Windows (PowerShell):

```bash
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest -v
```

## License

Apache-2.0 License

---

## Author

**Abhijith Sreekumar**
📧 [nbabhisreekumar@gmail.com]

> "A table-top toy robot simulation project."
