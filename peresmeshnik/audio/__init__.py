import os

PROCESSED = ('NVIDIA Broadcast', 'Krisp')  # virtual inputs that are the user's own microphone, its noise taken out


# outputs nobody hears: the call's devices (BlackHole, VB-Cable's input) and the call apps' own virtual speakers. Other
# virtual outputs play on to a real device (eqMac, Boom 3D, Background Music, Loopback, VoiceMeeter Input): allowed
SILENT = ('BlackHole', 'ZoomAudioDevice', 'Microsoft Teams Audio', 'VB-Audio Virtual Cable')


def silent_output(name: str) -> bool:
    """An output whose sound goes nowhere you hear (BlackHole 2ch / 16ch, CABLE Input, Zoom's or Teams' speaker)."""
    n = name.lower()
    return n.startswith('cable') or any(s.lower() in n for s in SILENT)


def processed(name: str) -> bool:
    """A virtual input that is the real microphone processed (NVIDIA Broadcast, Krisp): the user's voice, allowed."""
    return any(p.lower() in name.lower() for p in PROCESSED)  # as the native helpers match it: any case


def c_locale() -> dict:
    """The environment for a `pactl` whose output is read: pactl translates its messages (on a Russian desktop
    "Event 'new' on sink-input #5" reads "Событие «появление» в отношении входа аудиоприёмника №5")."""
    return {**os.environ, 'LC_ALL': 'C'}
