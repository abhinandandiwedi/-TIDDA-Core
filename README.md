# TIDDA — Tactical Intelligent Drone Defense Architecture

> **One operator. One command interface. Multiple autonomous agents.**

A software-first command & control platform for autonomous drone swarms and distributed tactical nodes.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--Time-orange)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# Overview

TIDDA (Tactical Intelligent Drone Defense Architecture) is a software-first Command & Control (C2) platform designed for autonomous drone swarms and distributed tactical nodes.

Instead of treating every drone as an isolated system, TIDDA treats the swarm as one coordinated platform that can be supervised by a single operator through a unified tactical interface.

The long-term objective is to reduce operator workload while improving situational awareness through distributed telemetry, autonomous coordination, AI-assisted perception, and scalable communication architecture.

The project follows a **simulation-first** development philosophy, allowing every subsystem to be validated in software before integration with real hardware.

> **Note:** TIDDA is a research and software engineering project. It does **not** implement autonomous weapon engagement. All threat information is intended to support a human operator.

---

# Current Features

## Swarm Simulation

- Multi-drone swarm simulator
- Autonomous waypoint navigation
- Return-to-base logic
- Low battery behavior
- Tactical map visualization
- Live telemetry updates

---

## Command & Control Backend

- FastAPI backend
- WebSocket communication
- Live telemetry aggregation
- Real-time dashboard broadcasting
- Modular backend architecture
- Automatic node management

---

## Mobile Node Architecture (NEW)

Android phones can now join TIDDA as real tactical nodes.

Current capabilities include:

- WebSocket connectivity
- Automatic node registration
- Heartbeat monitoring
- Live telemetry synchronization
- Automatic disconnect detection
- Multiple node support
- Zero frontend modifications required

The backend treats Android phones exactly like swarm nodes, allowing future expansion toward drones, robots, edge AI devices, and other tactical assets.

---

## Tactical Dashboard

React-based Ground Control Station featuring:

- Live tactical map
- Swarm visualization
- Node status
- Live telemetry
- Mission controls
- Event logging
- Real-time updates

---

# Current Architecture

```
                    Android Phone
                           │
                           │
                     WebSocket Client
                           │
                           ▼
┌─────────────────────────────────────────────┐
│             TIDDA Backend                   │
│---------------------------------------------│
│ Mobile Node Registry                        │
│ Swarm Simulation Engine                     │
│ Telemetry Aggregation                       │
│ Heartbeat Monitor                           │
│ WebSocket Server                            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          Tactical Dashboard (React)
                   │
                   ▼
             Human Operator
```

Future integrations

```
PX4
MAVLink
YOLO
SLAM
Mesh Networking
Distributed AI
```

---

# Engineering Goals

TIDDA investigates practical problems in:

- Multi-agent autonomy
- Distributed systems
- Swarm intelligence
- Human-machine teaming
- AI-assisted situational awareness
- Fault-tolerant communication
- Real-time robotics software
- Scalable command-and-control systems

---

# Current Development Progress

## Completed

- ✅ Swarm Simulator
- ✅ Tactical Dashboard
- ✅ FastAPI Backend
- ✅ WebSocket Infrastructure
- ✅ Live Telemetry
- ✅ Mobile Node Architecture
- ✅ Android Phone Integration
- ✅ Heartbeat System
- ✅ Automatic Node Registration
- ✅ Disconnect Detection
- ✅ Multi-node Communication Foundation

---

## In Progress

- Live GPS Streaming
- Battery Telemetry
- Camera Integration
- Sensor Data Collection
- Network Quality Monitoring

---

## Planned

- YOLO Detection
- Camera Streaming
- PX4 Integration
- MAVLink Support
- SLAM
- Mesh Networking
- Multi-Agent Mission Planning
- Sensor Fusion

---

# Project Structure

```
backend/
    api/
    websocket/
    mobile/
    swarm/
    simulator/
    dashboard/

frontend/
    React Dashboard

mobile/
    Android Tactical Node
```

---

# Roadmap

## Phase 1 ✅

- Swarm Simulator
- Tactical Dashboard
- Backend Architecture
- WebSocket Communication

## Phase 2 🚧

- Android Mobile Nodes
- GPS
- Camera
- Battery
- Telemetry

## Phase 3

- YOLO Integration
- Live Camera Feed
- Multi-phone Support
- Sensor Fusion

## Phase 4

- PX4
- MAVLink
- SLAM
- Mesh Networking
- Autonomous Mission Planning

---

# Screenshots

## Tactical Dashboard

> *(Add screenshot here)*

```
docs/images/dashboard.png
```

---

## Android Mobile Node

> *(Add screenshot here)*

```
docs/images/mobile_node.png
```

---

## Connected Mobile Node

> *(Add screenshot here)*

```
docs/images/mobile_connected.png
```

---

# Why TIDDA?

Most drone projects focus on controlling a single drone.

TIDDA focuses on controlling an entire autonomous system.

Instead of asking:

> "How do I fly one drone?"

TIDDA asks:

> "How can one operator command many autonomous agents through a single interface?"

This project is an exploration of scalable robotics software architecture where communication, coordination, and situational awareness become first-class engineering problems.

---

# Development Philosophy

TIDDA follows a **software-first** approach.

Every subsystem is designed, simulated, tested, and validated before hardware integration.

This allows:

- Faster iteration
- Better architecture
- Easier testing
- Hardware-independent development
- Cleaner software design

---

# Built With

- Python
- FastAPI
- React
- WebSockets
- OpenCV
- YOLO (planned integration)

---

# Author

**Abhinandan Diwedi**

Mechanical Engineering Student

Building software for autonomous drone swarm command-and-control systems.

---

# Project Status

🚧 **Active Development**

This project is under continuous development and serves as a long-term robotics software engineering initiative focused on autonomous systems and distributed command-and-control architectures.
