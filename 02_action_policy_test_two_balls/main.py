"""A small drag-and-drop target game built with pygame."""

from __future__ import annotations

import random

import pygame


WINDOW_SIZE = (900, 650)
FPS = 10

BACKGROUND = (8, 12, 24)
PLAY_AREA_COLOR = (15, 23, 42)
PLAY_AREA_BORDER = (71, 104, 145)
TEXT_COLOR = (232, 240, 255)
BLUE = (39, 174, 255)
BLUE_HOVER = (103, 213, 255)
RED = (255, 61, 94)
BUTTON_COLOR = (37, 150, 255)
BUTTON_HOVER = (75, 192, 255)
WHITE = (255, 255, 255)

PLAY_AREA = pygame.Rect(30, 90, WINDOW_SIZE[0] - 60, WINDOW_SIZE[1] - 120)
BLUE_RADIUS = 18
RED_RADIUS = 72
REPLAY_BUTTON = pygame.Rect(WINDOW_SIZE[0] // 2 - 75, 18, 150, 48)


def circles_intersect(
    first: pygame.Vector2,
    first_radius: int,
    second: pygame.Vector2,
    second_radius: int,
) -> bool:
    """Return whether two circles overlap or touch."""
    return first.distance_to(second) <= first_radius + second_radius


def overlap_score(
    blue_position: pygame.Vector2, red_position: pygame.Vector2
) -> int:
    """Map center distance within the overlap range to a score from 1 to 100."""
    maximum_distance = BLUE_RADIUS + RED_RADIUS
    distance = min(blue_position.distance_to(red_position), maximum_distance)
    return max(1, min(100, round(100 - 99 * distance / maximum_distance)))


def random_center(radius: int) -> pygame.Vector2:
    """Choose a center that keeps a circle entirely inside the play area."""
    return pygame.Vector2(
        random.randint(PLAY_AREA.left + radius, PLAY_AREA.right - radius),
        random.randint(PLAY_AREA.top + radius, PLAY_AREA.bottom - radius),
    )


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Blue Ball Target")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 38)
        self.text_font = pygame.font.Font(None, 27)
        self.button_font = pygame.font.Font(None, 30)

        self.blue_position = pygame.Vector2()
        self.red_position = pygame.Vector2()
        self.dragging = False
        self.won = False
        self.score = 0
        self.reset_round()

    def reset_round(self) -> None:
        """Place both balls randomly, with no overlap at the start."""
        self.red_position = random_center(RED_RADIUS)
        self.blue_position = random_center(BLUE_RADIUS)

        # The play area is much larger than the circles, so a valid position is
        # normally found immediately. The loop avoids beginning in a win state.
        while circles_intersect(
            self.blue_position, BLUE_RADIUS, self.red_position, RED_RADIUS
        ):
            self.blue_position = random_center(BLUE_RADIUS)

        self.dragging = False
        self.won = False
        self.score = 0

    def blue_contains(self, position: tuple[int, int]) -> bool:
        return self.blue_position.distance_to(position) <= BLUE_RADIUS

    def clamp_blue_to_play_area(self, position: tuple[int, int]) -> None:
        """Move the blue ball to the cursor while keeping it fully in bounds."""
        x = max(
            PLAY_AREA.left + BLUE_RADIUS,
            min(position[0], PLAY_AREA.right - BLUE_RADIUS),
        )
        y = max(
            PLAY_AREA.top + BLUE_RADIUS,
            min(position[1], PLAY_AREA.bottom - BLUE_RADIUS),
        )
        self.blue_position.update(x, y)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.won and REPLAY_BUTTON.collidepoint(event.pos):
                self.reset_round()
            elif not self.won and self.blue_contains(event.pos):
                self.dragging = True
                self.clamp_blue_to_play_area(event.pos)

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.clamp_blue_to_play_area(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.clamp_blue_to_play_area(event.pos)
                self.dragging = False
                if circles_intersect(
                    self.blue_position, BLUE_RADIUS, self.red_position, RED_RADIUS
                ):
                    self.score = overlap_score(self.blue_position, self.red_position)
                    self.won = True

        return True

    def draw_centered_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        center: tuple[int, int],
    ) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=center))

    def draw(self) -> None:
        mouse_position = pygame.mouse.get_pos()
        self.screen.fill(BACKGROUND)
        pygame.draw.rect(self.screen, PLAY_AREA_COLOR, PLAY_AREA)
        pygame.draw.rect(self.screen, PLAY_AREA_BORDER, PLAY_AREA, width=2)

        pygame.draw.circle(self.screen, RED, self.red_position, RED_RADIUS)

        blue_color = (
            BLUE_HOVER
            if not self.won and (self.dragging or self.blue_contains(mouse_position))
            else BLUE
        )
        pygame.draw.circle(self.screen, blue_color, self.blue_position, BLUE_RADIUS)
        pygame.draw.circle(self.screen, WHITE, self.blue_position, BLUE_RADIUS, width=2)

        if self.won:
            self.draw_centered_text(
                f"You win! Score: {self.score}",
                self.title_font,
                TEXT_COLOR,
                (230, REPLAY_BUTTON.centery),
            )
            button_color = (
                BUTTON_HOVER if REPLAY_BUTTON.collidepoint(mouse_position) else BUTTON_COLOR
            )
            pygame.draw.rect(self.screen, button_color, REPLAY_BUTTON, border_radius=8)
            self.draw_centered_text(
                "Replay", self.button_font, WHITE, REPLAY_BUTTON.center
            )
        else:
            self.draw_centered_text(
                "Drag the blue ball onto the red ball",
                self.title_font,
                TEXT_COLOR,
                (WINDOW_SIZE[0] // 2, 42),
            )

        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
