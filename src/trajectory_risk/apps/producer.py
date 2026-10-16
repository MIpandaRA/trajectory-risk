"""Window 1: the live simulator. Publishes every tick over TCP.

    traj-producer --map T11 --autostart

The simulation uses a fixed 1/60 s step. Each frame runs however many steps
real time requires (normally one), so simulation time stays true even when a
frame is slow; it never mixes measured frame time into the physics.
"""
from __future__ import annotations

import argparse
import math
import os
import time

from ..protocol import hello
from ..sim import World, WorldConfig
from ..transport import DEFAULT_HOST, DEFAULT_PORT, LiveStateServer

W, H, PANEL = 800, 600, 60


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="T11")
    ap.add_argument("--num-vehicles", type=int, default=0, help="0 = one per route")
    ap.add_argument("--num-pedestrians", type=int, default=6)
    ap.add_argument("--crossing-bias", type=float, default=0.3,
                    help="chance a pedestrian's next target is across a road (0 = original behaviour)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--autostart", action="store_true", help="start running immediately (else press Space)")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this much simulated time")
    args = ap.parse_args(argv)

    world = World(WorldConfig(args.map, args.num_vehicles, args.num_pedestrians,
                              crossing_bias=args.crossing_bias, seed=args.seed))
    server = LiveStateServer(args.host, args.port, hello=hello(args.map, 1 / world.cfg.dt, "simulator"))

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    screen = None if args.headless else pygame.display.set_mode((W, H + PANEL))
    if screen is not None:
        pygame.display.set_caption("Window 1 - simulator (producer)")
    font = pygame.font.Font(None, 20)
    running = args.autostart or args.headless
    clock = pygame.time.Clock()
    sim_clock_start = time.perf_counter()
    sim_time_at_start = 0.0

    try:
        while True:
            clock.tick(60)
            server.accept_pending()
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    running = not running
                    sim_clock_start, sim_time_at_start = time.perf_counter(), world.t_s
            if running:
                target_t = sim_time_at_start + (time.perf_counter() - sim_clock_start)
                steps = 0
                while world.t_s + world.cfg.dt <= target_t and steps < 10:  # cap catch-up
                    world.step()
                    server.broadcast(world.tick().to_dict())
                    steps += 1
                if steps == 10:  # far behind (debugger, sleep): resync instead of fast-forwarding
                    sim_clock_start, sim_time_at_start = time.perf_counter(), world.t_s
                if args.max_seconds and world.t_s >= args.max_seconds:
                    return
            if screen is not None:
                _draw(screen, world, font, running, server.n_clients)
    finally:
        server.close()
        pygame.quit()


def _draw(screen, world, font, running, n_clients):
    import pygame
    screen.fill((0, 0, 0))
    for v in world.vehicles:
        body = pygame.Surface((22, 14), pygame.SRCALPHA)
        pygame.draw.rect(body, (0, 200, 200), (0, 0, 22, 14), border_radius=5)
        rot = pygame.transform.rotate(body, -math.degrees(v.heading))
        screen.blit(rot, rot.get_rect(center=(v.x, v.y)))
    for p in world.pedestrians:
        pygame.draw.circle(screen, (235, 90, 205), (int(p.x), int(p.y)), 5)
    pygame.draw.rect(screen, (25, 25, 25), (0, H, W, PANEL))
    lines = [f"t {world.t_s:7.1f} s   vehicles {len(world.vehicles)}   pedestrians {len(world.pedestrians)}   "
             f"{'RUNNING' if running else 'PAUSED (Space)'}",
             f"consumers connected: {n_clients}"]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (200, 200, 200)), (14, H + 10 + 22 * i))
    pygame.display.flip()


if __name__ == "__main__":
    main()
