"""A small drag-and-drop target game built with pygame."""

from __future__ import annotations

import queue
import random
import threading

import pygame

from policy import PolicyController, TrainingResult, load_latest_policy, train_latest_policy
from training_data import TrainingDataCollector


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
RESET_BUTTON = pygame.Rect(WINDOW_SIZE[0] // 2 - 75, 18, 150, 48)
MENU_BUTTON = pygame.Rect(WINDOW_SIZE[0] - 145, 18, 115, 48)
MENU_PANEL = pygame.Rect(WINDOW_SIZE[0] - 310, 74, 280, 184)
COLLECTION_PANEL = pygame.Rect(WINDOW_SIZE[0] - 310, 74, 280, 184)
COLLECT_DATA_BUTTON = pygame.Rect(MENU_PANEL.x + 10, MENU_PANEL.y + 10, 260, 48)
TRAIN_MODEL_BUTTON = pygame.Rect(MENU_PANEL.x + 10, MENU_PANEL.y + 68, 260, 48)
RUN_MODE_BUTTON = pygame.Rect(MENU_PANEL.x + 10, MENU_PANEL.y + 126, 260, 48)
NEW_COLLECTION_BUTTON = pygame.Rect(
    COLLECTION_PANEL.x + 10, COLLECTION_PANEL.y + 10, 260, 48
)
CONTINUE_COLLECTION_BUTTON = pygame.Rect(
    COLLECTION_PANEL.x + 10, COLLECTION_PANEL.y + 68, 260, 48
)
BACK_BUTTON = pygame.Rect(
    COLLECTION_PANEL.x + 10, COLLECTION_PANEL.y + 126, 260, 48
)


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
        self.data_collector = TrainingDataCollector()
        self.menu_open = False
        self.menu_page = "main"
        self.training_thread: threading.Thread | None = None
        self.training_results: queue.SimpleQueue[
            tuple[TrainingResult | None, str | None]
        ] = queue.SimpleQueue()
        self.training_status = ""
        self.run_mode = False
        self.model_picked_up = False
        self.run_mode_score: int | None = None
        self.run_mode_replays = 0
        self.policy_controller: PolicyController | None = None
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
            if MENU_BUTTON.collidepoint(event.pos):
                self.menu_open = not self.menu_open
                self.menu_page = "main"
            elif self.menu_open and self.menu_page == "main":
                if COLLECT_DATA_BUTTON.collidepoint(event.pos):
                    self.menu_page = "collection"
                elif TRAIN_MODEL_BUTTON.collidepoint(event.pos):
                    self.menu_open = False
                    self.begin_model_training()
                elif RUN_MODE_BUTTON.collidepoint(event.pos):
                    self.menu_open = False
                    self.begin_run_mode()
                else:
                    self.menu_open = False
            elif self.menu_open and self.menu_page == "collection":
                if NEW_COLLECTION_BUTTON.collidepoint(event.pos):
                    self.start_data_collection(continue_latest=False)
                elif CONTINUE_COLLECTION_BUTTON.collidepoint(event.pos):
                    self.start_data_collection(continue_latest=True)
                elif BACK_BUTTON.collidepoint(event.pos):
                    self.menu_page = "main"
                else:
                    self.menu_open = False
            elif self.menu_open:
                self.menu_open = False
            elif self.run_mode and RESET_BUTTON.collidepoint(event.pos):
                self.run_mode = False
                self.policy_controller = None
                self.model_picked_up = False
                self.run_mode_score = None
                self.run_mode_replays = 0
                self.reset_round()
                self.training_status = "Run Mode stopped"
            elif self.won and REPLAY_BUTTON.collidepoint(event.pos):
                self.reset_round()
                if self.data_collector.active:
                    self.data_collector.begin_round(
                        self.blue_position, self.red_position, self.dragging
                    )
            elif not self.run_mode and not self.won and self.blue_contains(event.pos):
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

    def begin_run_mode(self) -> None:
        """Load the latest trained policy and give it control of the blue ball."""
        if self.data_collector.active:
            self.training_status = "Finish data collection before using Run Mode"
            return
        try:
            self.policy_controller = load_latest_policy()
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            self.training_status = f"Run Mode failed: {error}"
            return

        self.reset_round()
        self.model_picked_up = False
        self.run_mode_score = None
        self.run_mode_replays = 0
        self.run_mode = True
        self.training_status = f"Running {self.policy_controller.weight_path.name}"

    def run_policy_tick(self) -> None:
        """Apply one model-predicted action to the blue ball."""
        if not self.run_mode or self.policy_controller is None:
            return
        dx, dy, pick_up = self.policy_controller.action(
            (
                self.blue_position.x,
                self.blue_position.y,
                self.red_position.x,
                self.red_position.y,
                float(self.model_picked_up),
            )
        )
        was_picked_up = self.model_picked_up
        next_picked_up = pick_up >= 0.5
        if was_picked_up or next_picked_up:
            self.clamp_blue_to_play_area(
                (self.blue_position.x + dx, self.blue_position.y + dy)
            )
        self.model_picked_up = next_picked_up

        if not was_picked_up and next_picked_up:
            self.run_mode_score = None
        elif was_picked_up and not next_picked_up:
            if circles_intersect(
                self.blue_position, BLUE_RADIUS, self.red_position, RED_RADIUS
            ):
                self.run_mode_score = overlap_score(
                    self.blue_position, self.red_position
                )
                self.run_mode_replays += 1
                self.reset_round()
                self.model_picked_up = False
            else:
                self.run_mode_score = None

    def start_data_collection(self, continue_latest: bool) -> None:
        """Start a new data session or append a round to the newest session."""
        self.menu_open = False
        self.menu_page = "main"
        if self.data_collector.active:
            self.training_status = "Data collection is already active"
            return

        self.reset_round()
        try:
            if continue_latest:
                session_directory = self.data_collector.continue_latest_session(
                    self.blue_position, self.red_position, self.dragging
                )
                self.training_status = (
                    f"Continuing {session_directory.name} | round "
                    f"{self.data_collector.completed_rounds + 1}"
                )
            else:
                session_directory = self.data_collector.begin_session(
                    self.blue_position, self.red_position, self.dragging
                )
                self.training_status = (
                    f"Recording {session_directory.name} | round "
                    f"{self.data_collector.completed_rounds + 1}"
                )
        except (FileNotFoundError, OSError) as error:
            self.training_status = f"Collection failed: {error}"

    def begin_model_training(self) -> None:
        """Start policy training without blocking the Pygame event loop."""
        if self.training_thread is not None and self.training_thread.is_alive():
            self.training_status = "Model training is already running"
            return

        self.training_status = "Training model..."

        def train() -> None:
            try:
                result = train_latest_policy()
                self.training_results.put((result, None))
            except Exception as error:
                self.training_results.put((None, str(error)))

        self.training_thread = threading.Thread(target=train, daemon=True)
        self.training_thread.start()

    def check_training_result(self) -> None:
        """Apply a completed worker result on the main UI thread."""
        try:
            result, error = self.training_results.get_nowait()
        except queue.Empty:
            return

        if error is not None:
            self.training_status = f"Training failed: {error}"
        elif result is not None:
            self.training_status = (
                f"Saved {result.weight_path.name} | {result.samples} samples | "
                f"loss {result.final_loss:.4f}"
            )

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
            if not self.won
            and (
                self.dragging
                or (self.run_mode and self.model_picked_up)
                or (not self.run_mode and self.blue_contains(mouse_position))
            )
            else BLUE
        )
        pygame.draw.circle(self.screen, blue_color, self.blue_position, BLUE_RADIUS)
        pygame.draw.circle(self.screen, WHITE, self.blue_position, BLUE_RADIUS, width=2)

        if self.run_mode:
            run_mode_text = "Run Mode"
            if self.run_mode_score is not None:
                run_mode_text += f" | Score: {self.run_mode_score}"
            self.draw_centered_text(
                run_mode_text,
                self.title_font,
                TEXT_COLOR,
                (220, RESET_BUTTON.centery),
            )
            reset_color = (
                BUTTON_HOVER
                if RESET_BUTTON.collidepoint(mouse_position)
                else BUTTON_COLOR
            )
            pygame.draw.rect(self.screen, reset_color, RESET_BUTTON, border_radius=8)
            self.draw_centered_text(
                "Reset", self.button_font, WHITE, RESET_BUTTON.center
            )
        elif self.won:
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

        menu_color = (
            BUTTON_HOVER if MENU_BUTTON.collidepoint(mouse_position) else BUTTON_COLOR
        )
        pygame.draw.rect(self.screen, menu_color, MENU_BUTTON, border_radius=8)
        self.draw_centered_text("Menu", self.button_font, WHITE, MENU_BUTTON.center)

        if self.data_collector.active:
            rounds_text = f"Rounds recorded: {self.data_collector.completed_rounds}"
            rounds_surface = self.text_font.render(rounds_text, True, TEXT_COLOR)
            self.screen.blit(rounds_surface, (PLAY_AREA.left + 12, PLAY_AREA.top + 10))

        if self.training_status:
            status_text = self.training_status
            if len(status_text) > 72:
                status_text = status_text[:72] + "..."
            status_surface = self.text_font.render(status_text, True, TEXT_COLOR)
            self.screen.blit(
                status_surface,
                (PLAY_AREA.left + 12, PLAY_AREA.bottom - status_surface.get_height() - 10),
            )

        if self.menu_open:
            panel = COLLECTION_PANEL if self.menu_page == "collection" else MENU_PANEL
            pygame.draw.rect(self.screen, BACKGROUND, panel, border_radius=8)
            pygame.draw.rect(
                self.screen, PLAY_AREA_BORDER, panel, width=2, border_radius=8
            )
            if self.menu_page == "collection":
                self.draw_menu_button(
                    NEW_COLLECTION_BUTTON,
                    "Create New Session",
                    mouse_position,
                    enabled=not self.data_collector.active,
                )
                self.draw_menu_button(
                    CONTINUE_COLLECTION_BUTTON,
                    "Continue Latest Session",
                    mouse_position,
                    enabled=not self.data_collector.active,
                )
                self.draw_menu_button(
                    BACK_BUTTON, "Back", mouse_position, enabled=True
                )
            else:
                self.draw_menu_button(
                    COLLECT_DATA_BUTTON,
                    "Collection Active"
                    if self.data_collector.active
                    else "Training Data Collection",
                    mouse_position,
                    enabled=not self.data_collector.active,
                )
                training_running = (
                    self.training_thread is not None
                    and self.training_thread.is_alive()
                )
                self.draw_menu_button(
                    TRAIN_MODEL_BUTTON,
                    "Training..." if training_running else "Begin Training",
                    mouse_position,
                    enabled=not training_running,
                )
                self.draw_menu_button(
                    RUN_MODE_BUTTON,
                    "Run Mode",
                    mouse_position,
                    enabled=not self.data_collector.active,
                )

        pygame.display.flip()

    def draw_menu_button(
        self,
        rectangle: pygame.Rect,
        label: str,
        mouse_position: tuple[int, int],
        enabled: bool,
    ) -> None:
        if not enabled:
            color = (52, 67, 88)
        elif rectangle.collidepoint(mouse_position):
            color = BUTTON_HOVER
        else:
            color = BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rectangle, border_radius=6)
        self.draw_centered_text(label, self.text_font, WHITE, rectangle.center)

    def run(self) -> None:
        running = True
        while running:
            self.check_training_result()
            was_won = self.won
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            self.run_policy_tick()

            if self.data_collector.recording_round:
                self.data_collector.record_tick(
                    self.blue_position, self.red_position, self.dragging
                )
                if self.won and not was_won:
                    self.data_collector.finish_round()
                    completed_rounds = self.data_collector.completed_rounds
                    self.reset_round()
                    self.data_collector.begin_round(
                        self.blue_position, self.red_position, self.dragging
                    )
                    self.training_status = (
                        f"Recorded round {completed_rounds} | recording round "
                        f"{completed_rounds + 1}"
                    )

            self.draw()
            self.clock.tick(FPS)

        self.data_collector.close()
        pygame.quit()


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
