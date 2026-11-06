"""Simulation additions of the obstacle task (step 9): static cylinders, robot-obstacle contact and clearance.

spawn_obstacles() creates every cylinder of a scenario as a fixed body (mass 0) standing on the floor. The collector
and the harness call it right after the shared setup (evaluate_policy.setup_simulation: base reset, fingers, model),
before the first simulation step, so both start from the same state. Robot-obstacle contacts are generated even
with the config's collision_enable 0, which only toggles the robot's self-collision.

Contact (plan section 3): the robot touches an obstacle when the contact manifold of the two bodies is non-empty,
i.e. getContactPoints(robot, obstacle) returns any point. Bullet adds a point within its collision margins, so the
manifold is non-empty from a small positive separation on, which depends on the link (measured 2026-11-04 by
bisection, static robot: wheels 3.5 mm, chassis mesh 2.9 mm, finger tips 0.8 mm; tests/test_obstacle_sim.py checks
1 cm / 2 mm apart = no contact, touching = contact). The points are those of the last stepSimulation, which Bullet
computes from the poses at the start of that step. Contact forces are never used: they are not physical against a
fixed body (7.5 kN measured when pushing a wall).

Clearance: the smallest robot-obstacle surface distance over every robot link (chassis, wheels, arm, fingers) from
getClosestPoints; negative when penetrating, inf without obstacles. All helpers only query the simulation.
"""
import math

import pybullet as p

OBSTACLE_RGBA = [0.75, 0.25, 0.15, 1.0]
CLEARANCE_SEARCH = 100.0   # m, getClosestPoints search distance (beyond the task's extent)


def spawn_obstacles(client, obstacles):
    """Fixed cylinders from [{x, y, r, h}, ...] standing on the floor; returns their body ids."""
    bodies = []
    for ob in obstacles:
        radius, height = float(ob["r"]), float(ob["h"])
        collision = client.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        visual = client.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=OBSTACLE_RGBA)
        bodies.append(client.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                             baseVisualShapeIndex=visual,
                                             basePosition=[float(ob["x"]), float(ob["y"]), height / 2]))
    return bodies


def robot_obstacle_contact(client, robot_id, obstacle_ids):
    """(True, robot link index of the first contact point) if the robot touches any obstacle, else (False, None).
    Link -1 is the robot's base link. Query it after a simulation step."""
    for body_id in obstacle_ids:
        contacts = client.getContactPoints(bodyA=robot_id, bodyB=body_id)
        if contacts:
            return True, int(contacts[0][3])
    return False, None


def link_name(client, body_id, link):
    """Name of a link of a body (link -1 = the base link), e.g. 'robot_j2s7s300_link_finger_tip_1'."""
    if link == -1:
        return client.getBodyInfo(body_id)[0].decode()
    return client.getJointInfo(body_id, link)[12].decode()


def min_clearance(client, robot_id, obstacle_ids, distance=CLEARANCE_SEARCH):
    """Smallest robot-obstacle surface distance (m) over all robot links; negative = penetration, inf if none."""
    closest = math.inf
    for body_id in obstacle_ids:
        points = client.getClosestPoints(bodyA=robot_id, bodyB=body_id, distance=distance)
        if points:
            closest = min(closest, min(float(point[8]) for point in points))
    return closest
