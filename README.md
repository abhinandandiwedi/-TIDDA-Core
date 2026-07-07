# TIDDA — Tactical Intelligent Drone Defense Architecture

**An autonomous drone swarm command & control platform for surveillance, threat detection, and coordinated multi-agent operations.**

Built by Abhinandan Diwedi

---

## Overview

TIDDA is a full-stack command-and-control (C2) system for autonomous drone swarms, designed around a simple principle: **an operator should not need to manually monitor every drone individually.** As swarm size scales, cognitive load on a human operator becomes the real bottleneck — not hardware. TIDDA's core value is reducing that load through autonomous coordination, self-healing communication, and AI-assisted threat detection, while keeping critical decisions human-in-the-loop.

This repository contains the current software/simulation stack — the swarm intelligence engine, the C2 backend, the live tactical dashboard, and the computer vision detection pipeline.

> **Note on scope:** TIDDA is built and demonstrated as a software/simulation platform. It does not involve physical weapons deployment or autonomous engagement — all threat detection and response actions are surfaced to a human operator for decision-making.

---

## Core Capabilities

### 🧠 Swarm Intelligence Engine
- Multi-drone coordination with autonomous behaviors: automatic Return-to-Base on low battery, threat evasion, and stealth-perch (power-conserving standby) maneuvers
- Self-healing mesh network logic — if one node loses signal, data reroutes automatically through neighboring nodes rather than dropping
- Lightweight, dependency-free simulation core so the system can be demonstrated without external flight-simulator infrastructure

### 📡 Command & Control Backend
- FastAPI + WebSocket server aggregating live telemetry from all connected drones/nodes
- Real-time broadcast architecture to the frontend dashboard
- Modular design supporting both simulated drones and real MAVLink-based telemetry (ArduPilot SITL compatible)

### 🎯 AI-Assisted Threat Detection
- YOLOv8-based object detection pipeline processing live camera feeds
- Detections are pushed to the operator dashboard in real time as actionable alerts
- Designed as a decision-support layer — flags what a human should look at, rather than acting autonomously

### 🖥️ Tactical Dashboard (GCS UI)
- Live tactical map showing drone positions, status, and alert history
- Multi-node view designed to scale as swarm size grows
- Built with React for real-time responsiveness

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   Drone / Sensor Layer   │         │   Detection Layer         │
│  (Simulated or MAVLink)  │         │  (YOLOv8 + Camera Feed)   │
└────────────┬─────────────┘         └────────────┬─────────────┘
             │                                     │
             └───────────────┬─────────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │     C2 Backend (FastAPI)       │
              │  Swarm Logic · Mesh Relay ·     │
              │  Telemetry Aggregation          │
              └───────────────┬─────────────────┘
                              │ WebSocket
                              ▼
              ┌───────────────────────────────┐
              │   Tactical Dashboard (React)   │
              │  Live Map · Alerts · Status     │
              └───────────────────────────────┘
```

---

## Why This Matters

Most drone software focuses on flying a single vehicle well. TIDDA is built around the harder problem: **coordinating many drones as one system**, and making sure that system stays operational and legible to a human even when individual nodes fail, lose signal, or run low on power. That's the gap this project targets — turning a swarm from "many things to babysit" into "one system to command."

---

## Status

Actively in development. This is a continuously evolving platform, not a one-off build — features and modules here are being iterated on across multiple development cycles and competitions.

---

## Roadmap

- Expanded multi-drone real-world telemetry (beyond simulation)
- Threat scoring and sensor fusion across multiple detection sources
- Indoor positioning and mapping capabilities
- Physical hardware platform (long-term)

---

## License

