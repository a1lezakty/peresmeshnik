import os


def c_locale() -> dict:
    """The environment for a `pactl` whose output is read: pactl translates its messages (on a Russian desktop
    "Event 'new' on sink-input #5" reads "Событие «появление» в отношении входа аудиоприёмника №5")."""
    return {**os.environ, 'LC_ALL': 'C'}
