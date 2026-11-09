import os

PROCESSED = ('NVIDIA Broadcast', 'Krisp')  # virtual inputs that are the user's own microphone, its noise taken out


def processed(name: str) -> bool:
    """A virtual input that is the real microphone processed (NVIDIA Broadcast, Krisp): the user's voice, allowed."""
    return any(p.lower() in name.lower() for p in PROCESSED)


def c_locale() -> dict:
    """The environment for a `pactl` whose output is read: pactl translates its messages (on a Russian desktop
    "Event 'new' on sink-input #5" reads "Событие «появление» в отношении входа аудиоприёмника №5")."""
    return {**os.environ, 'LC_ALL': 'C'}
