"""Voice enrollment: a 30-60 s recording of the user's own voice -> a voice profile on disk.

    python -m peresmeshnik.voice enroll --record 45 --lang ru     # read the prompted text aloud
    python -m peresmeshnik.voice enroll --import me.wav --lang ru
    python -m peresmeshnik.voice check me.wav
    python -m peresmeshnik.voice list

A profile is a folder `<voices>/<id>/` with `voice.json` (schema v2 below). My own voice (kind 'me') also has
`reference.wav` (level-normalised); clone engines add their own cache next to it (XTTS: `xtts.pt`). Profiles stay on
this computer (my voice goes only to my own server, for a session). This module is torch-free; the clone engine is
imported only for `--engine`.

Two checks, each with a verdict for the user ("good" / "record again, because ..."): `check()` of the
recording (length, speech time, level, clipping, background), and after cloning `check_clone()`: the
clone says test phrases in the language it will speak, the app's ASR must understand them (WER) and,
where the engine can tell, they must sound like the recording (speaker similarity). Thresholds: BENCH.

Whose voice it is (M-20): `confirm_live()` - two random phrases read at the microphone against every window of the
recording, and the recording's start against its own rest; then `confirm_clone()` / `vet_clone()` - the clone made from
it says a test phrase, heard against that live reading. `mine()` / `why_not_mine()`: the one gate of "speaks as me".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from peresmeshnik.engines import Voice

STOCK = 'piper'  # [local] clone = this: me -> them speaks in the stock Piper voice of its language, not a clone of mine
MIN_S, MAX_S = 30.0, 60.0  # recording length
MIN_SPEECH_S = 15.0  # speech inside it (pauses and silence excluded)
MIN_SR = 16_000
CLIP = 0.99  # |sample| at or above counts as clipped
MAX_CLIPPED = 0.001  # share of clipped samples
QUIET_DB, LOW_DB = -45.0, -35.0  # speech level, dBFS: fail below QUIET_DB, warn below LOW_DB
NOISY_DB, FAIR_DB = 12.0, 20.0  # speech above the background, dB: fail / warn below (a LibriVox reader: 16)
TARGET_DB = -20.0  # saved reference: speech level, peak kept under -1 dBFS
BAND_DB = -45.0  # speech energy at 4-8 kHz vs 0.1-4 kHz: below = a telephone-band mic (BENCH: phone -74..-80, mics -27..-9)

PROMPTS = {  # read aloud for 30-45 s; ru, en: written for the app (varied sounds, natural pauses at punctuation);
    # the others: public-domain prose, verbatim from the source (author, work, year; docs/MODELS.md "Texts")
    'ru': 'Привет! Я записываю этот короткий текст, чтобы приложение узнало, как звучит мой голос. '
          'Неделя выдалась насыщенной. В понедельник шёл дождь, и весь день пришлось работать дома. '
          'Во вторник мы встретились с друзьями, пили чай с вареньем и спорили о кино. '
          'В среду мы съездили за город: там тихо, пахнет хвоей, а у озера щебечут птицы. '
          'Четверг и пятница пролетели незаметно — письма, звонки, короткие встречи. '
          'А в выходные наконец-то удалось выспаться, потом был вкусный ужин и долгий разговор с родителями. Спасибо!',
    'en': 'Hello! I am recording this short passage so the app can learn how my voice sounds. '
          'Last week was busy but pleasant. On Monday it rained, so I worked from home all day. '
          'On Tuesday I met a few friends for coffee, and we argued about films for hours. '
          'On Wednesday I drove out of town: it was quiet, the air smelled of pine, and birds sang by the lake. '
          'Thursday and Friday flew by with emails, phone calls and quick meetings. '
          'At the weekend I finally slept in, cooked a proper dinner and called my parents. Thanks for listening!',
    # Михайло Коцюбинський (1913), Intermezzo, 1909
    # https://uk.wikisource.org/w/index.php?title=Твори_(Коцюбинський,_1955)/2/Intermezzo&action=render
    'uk': "Ніколи перше не почував я так ясно зв'язку з землею, як тут. В городах земля одягнена в камінь і залізо і "
          'недоступна. Тут я став близький до неї. Свіжими ранками я перший будив сонну ще воду криниці. Коли порожнє відро '
          'плескалось денцем об її груди, вона ухала гучно спросоння у глибині й ліниво вливалась у нього. Потому тремтіла, '
          'сиза на сонці. Я пив її свіжу, холодну, ще повну снів, і хлюпав нею собі в лице. Після того було молоко.',
    # Franz Kafka (1924), Die Verwandlung, 1915
    # https://www.gutenberg.org/cache/epub/22367/pg22367.txt
    'de': 'Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem Bett zu einem ungeheueren '
          'Ungeziefer verwandelt. Er lag auf seinem panzerartig harten Rücken und sah, wenn er den Kopf ein wenig hob, seinen '
          'gewölbten, braunen, von bogenförmigen Versteifungen geteilten Bauch, auf dessen Höhe sich die Bettdecke, zum '
          'gänzlichen Niedergleiten bereit, kaum noch erhalten konnte. Seine vielen, im Vergleich zu seinem sonstigen Umfang '
          'kläglich dünnen Beine flimmerten ihm hilflos vor den Augen. »Was ist mit mir geschehen?« dachte er. Es war kein '
          'Traum.',
    # Alphonse Daudet (1897), Lettres de mon moulin (Installation), 1869
    # https://www.gutenberg.org/cache/epub/36780/pg36780.txt
    'fr': "C'est de là que je vous écris, ma porte grande ouverte, au bon soleil. Un joli bois de pins tout étincelant de "
          "lumière dégringole devant moi jusqu'au bas de la côte. A l'horizon, les Alpilles découpent leurs crêtes fines... "
          'Pas de bruit... A peine, de loin en loin, un son de fifre, un courlis dans les lavandes, un grelot de mules sur la '
          'route... Tout ce beau paysage provençal ne vit que par la lumière. Et maintenant, comment voulez-vous que je le '
          'regrette, votre Paris bruyant et noir? Je suis si bien dans mon moulin!',
    # Horacio Quiroga (1937), Cuentos de la selva: La tortuga gigante, 1918
    # https://es.wikisource.org/w/index.php?title=La_tortuga_gigante&action=render
    'es': 'Vivía solo en el bosque, y él mismo se cocinaba. Comía pájaros y bichos del monte, que cazaba con la escopeta, y '
          'después comía frutas. Dormía bajo los árboles, y cuando hacía mal tiempo construía en cinco minutos una ramada con '
          'hojas de palmera, y allí pasaba sentado y fumando, muy contento en medio del bosque que bramaba con el viento y la '
          'lluvia. Había hecho un atado con los cueros de los animales, y lo llevaba al hombro.',
    # Carlo Collodi (1890), Le avventure di Pinocchio, 1883
    # https://www.gutenberg.org/cache/epub/52484/pg52484.txt
    'it': "C'era una volta un pezzo di legno. Non era un legno di lusso, ma un semplice pezzo da catasta, di quelli che "
          "d'inverno si mettono nelle stufe e nei caminetti per accendere il fuoco e per riscaldare le stanze. Non so come "
          'andasse, ma il fatto gli è che un bel giorno questo pezzo di legno capitò nella bottega di un vecchio falegname, il '
          "quale aveva nome mastr'Antonio, se non che tutti lo chiamavano maestro Ciliegia, per via della punta del suo naso, "
          'che era sempre lustra e paonazza, come una ciliegia matura.',
    # Machado de Assis (1908), Conto de Escola (Várias Histórias), 1884
    # https://pt.wikisource.org/w/index.php?title=Conto_de_Escola&action=render
    'pt': 'Com franqueza, estava arrependido de ter vindo. Agora que ficava preso, ardia por andar lá fora, e recapitulava o '
          'campo e o morro, pensava nos outros meninos vadios, o Chico Telha, o Américo, o Carlos das Escadinhas, a fina flor '
          'do bairro e do gênero humano. Para cúmulo de desespero, vi através das vidraças da escola, no claro azul do céu, '
          'por cima do morro do Livramento, um papagaio de papel, alto e largo, preso de uma corda imensa, que bojava no ar, '
          'uma cousa soberba.',
    # Henryk Sienkiewicz (1916), Latarnik, 1881
    # https://wolnelektury.pl/media/book/txt/latarnik.txt
    'pl': 'Tegoż samego jeszcze wieczora, gdy słońce stoczyło się na drugą stronę międzymorza, a po dniu promiennym nastąpiła '
          'noc bez zmierzchu, nowy latarnik widocznie był już na miejscu, bo latarnia rzuciła jak zwykle na wody swoje snopy '
          'jaskrawego światła. Noc była zupełnie spokojna, cicha, prawdziwie podzwrotnikowa, przesycona jasną mgłą, tworzącą '
          'koło księżyca wielki, zabarwiony tęczowo krąg o miękkich, nieujętych brzegach. Morze tylko burzyło się, ponieważ '
          'przypływ wzbierał. Skawiński stał na balkonie, tuż koło olbrzymich ognisk, podobny z dołu do małego, czarnego '
          'punkciku.',
    # Ömer Seyfettin (1920), Kaşağı, 1919
    # https://tr.wikisource.org/w/index.php?title=Kaşağı&action=raw
    'tr': 'Ahırın avlusunda oynarken aşağıda, gümüş söğütler altında görünmeyen derenin hazin şırıltısını işitirdik. Evimiz iç '
          "çitin büyük kestane ağaçları arkasında kaybolmuş gibiydi. Annem, İstanbul'a gittiği için benden bir yaş küçük olan "
          "kardeşim Hasan'la artık Dadaruh'un yanından hiç ayrılmıyorduk. Bu, babamın seyisi, ihtiyar bir adamdı. Sabahleyin "
          "erkenden ahıra koşuyorduk. En sevdiğimiz şey atlardı. Dadaruh'la beraber onları suya götürmek, çıplak sırtlarına "
          'binmek, ne doyulmaz bir zevkti. Hasan korkar, yalnız binemezdi. Dadaruh onu kendi önüne alırdı. Torbalara arpa '
          'koymak, yemliklere ot doldurmak, gübreleri kaldırmak eğlenceli bir oyundan ziyade, bizim hoşumuza gidiyordu.',
    # 鲁迅 (1936), 从百草园到三味书屋（朝花夕拾）, 1926
    # https://www.marxists.org/chinese/reference-books/luxun/06/008.htm
    'zh': '我家的后面有一个很大的园，相传叫作百草园。现在是早已并屋子一起卖给朱文公的子孙了，连那最末次的相见也已经隔了七八年，'
          '其中似乎确凿只有一些野草；但那时却是我的乐园。不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑椹；也不必说鸣蝉'
          '在树叶里长吟，肥胖的黄蜂伏在菜花上，轻捷的叫天子（云雀）忽然从草间直窜向云霄里去了。单是周围的短短的泥墙根一带，就有'
          '无限趣味。',
    # 芥川龍之介 (1927), トロッコ, 1922
    # https://www.aozora.gr.jp/cards/000879/files/43016_16836.html
    'ja': '或夕方、――それは二月の初旬だった。良平は二つ下の弟や、弟と同じ年の隣の子供と、トロッコの置いてある村外れへ行った。ト'
          'ロッコは泥だらけになったまま、薄明るい中に並んでいる。が、その外は何処を見ても、土工たちの姿は見えなかった。三人の子'
          '供は恐る恐る、一番端にあるトロッコを押した。トロッコは三人の力が揃うと、突然ごろりと車輪をまわした。良平はこの音にひ'
          'やりとした。しかし二度目の車輪の音は、もう彼を驚かさなかった。',
    # 이효석 (1942), 메밀꽃 필 무렵, 1936
    # https://ko.wikisource.org/w/index.php?title=메밀꽃_필_무렵&action=raw
    'ko': '밤중을 지난 무렵인지 죽은 듯이 고요한 속에서 짐승 같은 달의 숨소리가 손에 잡힐 듯이 들리며, 콩 포기와 옥수수 잎새가 '
          '한층 달에 푸르게 젖었다. 산허리는 온통 메밀 밭이어서 피기 시작한 꽃이 소금을 뿌린 듯이 흐붓한 달빛에 숨이 막힐 '
          '지경이다. 붉은 대공이 향기같이 애잔하고 나귀들의 걸음도 시원하다. 길이 좁은 까닭에 세 사람은 나귀를 타고 외줄로 '
          '늘어섰다. 방울소리가 시원스럽게 딸랑딸랑 메밀 밭께로 흘러간다.',
    # جبران خليل جبران (1931), الأجنحة المتكسرة — في باب الهيكل, 1912
    # https://ar.wikisource.org/w/index.php?title=الأجنحة_المتكسرة/في_باب_الهيكل&action=raw
    'ar': 'وبعد أيام وقد مللت الوحدة، وتعبت أجفاني من النظر إلى أوجه الكتب العابسة علوت مركبة طالباً منزل فارس كرامة، حتى إذا '
          'ما بلغت بي غابة الصنوبر حيث يذهب القوم للتنزه، حول السائق وجهة فرسيه عن الطريق العمومية فسار خبباً على ممر تظلله '
          'أشجار الصفصاف وتتمايل على جانبيه الأعشاب والدوالي المتعرشة وأزاهر نيسان المبتسمة بثغور حمراء كالياقوت وزرقاء '
          'كالزمرد وصفراء كالذهب.',
}


@dataclass
class Report:
    seconds: float
    speech_s: float
    level_db: float
    noise_db: float
    snr_db: float
    clipped: float
    band_db: float = 0.0  # speech energy above 4 kHz vs below
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _frames(x: np.ndarray, sr: int, ms: int = 30) -> np.ndarray:
    n = sr * ms // 1000
    return x[:len(x) // n * n].reshape(-1, n).astype(np.float64)


def _band_db(frames: np.ndarray, sr: int) -> float:
    """Speech energy at 4-8 kHz relative to 0.1-4 kHz (averaged spectrum of the speech frames)."""
    if not len(frames) or sr < 16_000:
        return 0.0
    n = frames.shape[1]
    p = np.mean(np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    return float(10 * np.log10(p[(f >= 4000) & (f < 8000)].sum() / (p[(f >= 100) & (f < 4000)].sum() + 1e-20) + 1e-12))


def check(pcm: np.ndarray, sr: int) -> Report:
    """Quality of an enrollment recording: length, speech time, level, clipping, noise."""
    x = np.asarray(pcm, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    raw, x = x, x - x.mean(dtype=np.float64) if len(x) else x  # a DC offset is no sound: not noise, not level
    frames = _frames(x, sr) if sr >= 100 and len(x) >= sr // 10 else np.zeros((0, 1))
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12) if len(frames) else np.array([-120.0])
    live = db[db > -100]  # digital silence (gaps, a muted mic) is no noise floor
    noise = float(np.percentile(live, 10)) if len(live) else -120.0  # pauses between phrases
    is_speech = db > max(noise + 6, -60)
    speech = db[is_speech]
    level = float(10 * np.log10(np.mean(10 ** (speech / 10)))) if len(speech) else -120.0
    r = Report(seconds=round(len(x) / sr, 2), speech_s=round(len(speech) * 0.03, 2), level_db=round(level, 1),
               noise_db=round(noise, 1), snr_db=round(level - noise, 1),
               clipped=round(float(np.mean(np.abs(raw) >= CLIP)) if len(x) else 0.0, 5),
               band_db=round(_band_db(frames[is_speech] if len(frames) else frames, sr), 1))
    if sr < MIN_SR:
        r.errors.append(f'sample rate {sr} Hz is below {MIN_SR} Hz')
    if r.seconds < MIN_S:
        r.errors.append(f'{r.seconds:.0f} s recorded, need {MIN_S:.0f}-{MAX_S:.0f} s')
    elif r.seconds > MAX_S:
        r.warnings.append(f'{r.seconds:.0f} s recorded, only the first {MAX_S:.0f} s are used')
    if r.speech_s < MIN_SPEECH_S:
        r.errors.append(f'only {r.speech_s:.0f} s of speech, need at least {MIN_SPEECH_S:.0f} s: talk more, pause less')
    if r.clipped > MAX_CLIPPED:
        r.errors.append(f'clipping in {r.clipped:.2%} of samples: lower the mic gain or move back a little')
    if level < QUIET_DB:
        r.errors.append(f'speech level {level:.0f} dBFS is too quiet: raise the mic gain or come closer')
    elif level < LOW_DB:
        r.warnings.append(f'speech level {level:.0f} dBFS is low')
    if r.snr_db < NOISY_DB:
        r.errors.append(f'speech is only {r.snr_db:.0f} dB above the background: find a quieter place')
    elif r.snr_db < FAIR_DB:
        r.warnings.append(f'background noise is noticeable ({r.snr_db:.0f} dB below speech)')
    if len(speech) and r.band_db < BAND_DB:
        r.warnings.append('no sound above 4 kHz, like a telephone or a Bluetooth headset microphone: the clone will sound '
                          'muffled; a wired or built-in microphone is better')
    return r


def verdict(r: Report) -> str:
    """One line for the user: good, good with a caveat, or record again and why."""
    if r.errors:
        return 'Record again: ' + '; '.join(r.errors) + '.'
    if r.warnings:
        return 'Good enough: ' + '; '.join(r.warnings) + '.'
    return 'Good: enough clear speech, no clipping, a quiet background.'


def summary(r: Report) -> str:
    return (f'length {r.seconds:.1f} s, speech {r.speech_s:.1f} s, level {r.level_db:.1f} dBFS, '
            f'noise {r.noise_db:.1f} dBFS (speech {r.snr_db:.1f} dB above it), clipped {r.clipped:.3%}, '
            f'above 4 kHz {r.band_db:.0f} dB')


def enroll(pcm: np.ndarray, sr: int, name: str = 'me', lang: str = 'ru', transcript: str | None = None,
           force: bool = False, root: Path | None = None, source: str = 'file') -> tuple[Report, Voice | None]:
    """Check a recording and save it as the profile `name` if it passes (or `force`): (report, profile or None).
    `source`: 'mic' recorded by the app, 'file' imported. Not confirmed as mine yet (confirm_live, set_consent)."""
    r = check(pcm, sr)
    return r, (save(pcm, sr, name, lang, transcript, root, source) if r.ok or force else None)


# --- after cloning: does the clone speak understandably, in the recorded voice? ---

TEST_PHRASES = {  # what the clone says in the check: everyday words, numbers spelled out, ~4 s each
    'en': ['Good morning, can you hear me well? I will be there in about ten minutes.',
           'Thank you for waiting. Let us start with the first question on the list.'],
    'ru': ['Добрый день, вы меня хорошо слышите? Я буду минут через десять.',
           'Спасибо, что подождали. Давайте начнём с первого вопроса в списке.'],
    'uk': ['Добрий день, ви мене добре чуєте? Я буду хвилин за десять.',
           'Дякую, що зачекали. Почнімо з першого питання у списку.'],
    'de': ['Guten Tag, können Sie mich gut hören? Ich bin in etwa zehn Minuten da.',
           'Danke fürs Warten. Fangen wir mit der ersten Frage auf der Liste an.'],
    'fr': ["Bonjour, vous m'entendez bien ? Je serai là dans une dizaine de minutes.",
           "Merci d'avoir attendu. Commençons par la première question de la liste."],
    'es': ['Buenos días, ¿me oye bien? Llegaré en unos diez minutos.',
           'Gracias por esperar. Empecemos con la primera pregunta de la lista.'],
    'it': ['Buongiorno, mi sente bene? Arrivo tra una decina di minuti.',
           'Grazie per aver aspettato. Cominciamo dalla prima domanda della lista.'],
    'pt': ['Bom dia, você está me ouvindo bem? Chego em uns dez minutos.',
           'Obrigado por esperar. Vamos começar pela primeira pergunta da lista.'],
    'pl': ['Dzień dobry, czy dobrze mnie słychać? Będę za jakieś dziesięć minut.',
           'Dziękuję za cierpliwość. Zacznijmy od pierwszego pytania na liście.'],
    'tr': ['İyi günler, beni iyi duyabiliyor musunuz? Yaklaşık on dakika içinde orada olurum.',
           'Beklediğiniz için teşekkürler. Listedeki ilk soruyla başlayalım.'],
    'zh': ['你好，你能听清楚我说话吗？我大概十分钟后到。', '谢谢你的耐心等待。我们从列表上的第一个问题开始吧。'],
    'ja': ['こんにちは、私の声はよく聞こえますか？十分ほどで着きます。', 'お待たせしました。リストの最初の質問から始めましょう。'],
    'ko': ['안녕하세요, 제 목소리 잘 들리세요? 십 분쯤 후에 도착해요.', '기다려 주셔서 감사합니다. 목록의 첫 번째 질문부터 시작하죠.'],
    'ar': ['مرحبًا، هل تسمعني جيدًا؟ سأصل بعد عشر دقائق تقريبًا.', 'شكرًا على الانتظار. لنبدأ بالسؤال الأول في القائمة.'],
}
MAX_WER, WARN_WER = 0.25, 0.10  # word error rate of the app's ASR on the clone's test phrases: fail / warn above
# speaker similarity of the clone to the recording (XTTS's own encoder, cross-lingual): fail / warn below. BENCH: the
# clone of the same speaker 0.37-0.68 (median 0.54) over clean and degraded recordings, of another speaker median 0.11
MIN_SIM, WARN_SIM = 0.25, 0.35
# engines whose similarity has no threshold (the runner's `kind` in VoicePackClone.info): only WER judges their clone.
# Qwen3-TTS: the cosine of its own x-vector, the very one its clone speaks from: its clone 0.96-0.99 against its own
# recording, another speaker's recording 0.87-0.97 (BENCH, bench/voice_check.py qwen3): no threshold tells them apart
NO_SIM = {'qwen3'}


@dataclass
class CloneReport:
    wer: float
    similarity: float | None  # None: the engine cannot tell
    heard: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _words(text: str) -> list[str]:
    """Words; in Chinese and Japanese (no spaces) each Han or kana character is one, so WER is a character error rate.
    Arabic vowel marks (tanwin in the test phrases) are dropped: an ASR writes none, and they would split a word."""
    import re
    han = '\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
    text = re.sub('[\u064b-\u0652\u0670]', '', text.lower().replace('ё', 'е'))
    return re.findall(rf'[{han}]|[^\W{han}]+', re.sub(r'[^\w\s]', ' ', text))


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance / reference words (case and punctuation ignored)."""
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def check_clone(tts, voice: Voice, lang: str, asr, phrases: list[str] | None = None) -> CloneReport:
    """The clone (`tts`, a `Tts` with the profile `voice`) says the test phrases in `lang`; `asr` (local/asr.OnnxAsr
    for that language) transcribes them. Similarity comes from `tts.similarity(voice, pcm, sr)` where it exists and has
    thresholds (not NO_SIM)."""
    from peresmeshnik.engines import SAMPLE_RATE
    phrases = phrases or TEST_PHRASES[lang]
    heard, audio = [], []
    for text in phrases:
        pcm = np.concatenate(list(tts.synth(text, lang, voice)) or [np.zeros(1, np.float32)])
        audio.append(pcm)
        x = np.interp(np.arange(0, len(pcm), tts.sample_rate / SAMPLE_RATE), np.arange(len(pcm)), pcm).astype(np.float32)
        heard.append(asr.recognize(x).text)
    err = wer(' '.join(phrases), ' '.join(heard))
    judged = hasattr(tts, 'similarity') and not (isinstance(i := getattr(tts, 'info', None), dict) and i.get('kind') in NO_SIM)
    sim = tts.similarity(voice, np.concatenate(audio), tts.sample_rate) if judged else None
    r = CloneReport(round(err, 3), None if sim is None else round(float(sim), 3), heard)
    if err > MAX_WER:
        r.errors.append(f'the clone is hard to understand ({err:.0%} of the words misheard)')
    elif err > WARN_WER:
        r.warnings.append(f'{err:.0%} of the words misheard')
    if sim is not None and sim < MIN_SIM:
        r.errors.append(f'the clone does not sound like the recording (similarity {sim:.2f})')
    elif sim is not None and sim < WARN_SIM:
        r.warnings.append(f'the clone sounds only somewhat like you (similarity {sim:.2f})')
    return r


def clone_verdict(r: CloneReport) -> str:
    sim = '' if r.similarity is None else f', similarity {r.similarity:.2f}'
    if r.errors:
        return (f'Record again: {"; ".join(r.errors)}. A quiet room, the usual distance to the mic, your normal voice '
                f'(WER {r.wer:.0%}{sim}).')
    if r.warnings:
        return f'Usable: {"; ".join(r.warnings)} (WER {r.wer:.0%}{sim}).'
    like = '' if r.similarity is None else ' and sounds like the recording'  # not claimed where nothing measured it
    return f'Good: the clone is understood{like} (WER {r.wer:.0%}{sim}).'


def voices_dir() -> Path:
    from peresmeshnik.config import data_dir
    return data_dir() / 'voices'


# --- voice.json v2: my voice and the voices of the people I talk to (RESEARCH-clone §5-6, FINDINGS M-20) ---
#   {version: 2, id, name, kind, lang, created, consent, seconds, engines,
#    and for kind 'me' as in v1: transcript, sample_rate, gain_db, quality}   (v1 readers still find their keys)
# id       the folder: 'me' ('me.new' while the wizard holds a new take); a contact 'them-<8 hex>': no name in a path
# name     shown on the voices screen; never in logs or crash reports
# kind     'me': may speak into the call (virtual mic) and go to my server; 'them': only ever into my headphones
# consent  None (not confirmed: it speaks nowhere as me) or {how, at, wer, similarity, windows, ref_sha256}; how:
#            'mic-live'          (me) recorded at the microphone, then two random phrases read there: the app's ASR
#                                heard them (wer), every window of the recording is the reader's voice (similarity: min)
#            'import+mic-match'  (me) a file, confirmed the same way
#            'call-saved'        (them) kept from a call by an explicit "remember", the other person agreed
#          ref_sha256: the reference.wav that was checked; another file there (a new take, a crash mid-save) is not
#          confirmed. A v1 profile has none: 0.3.0 checked nothing (FIXES-s3 2)
# source   (me) 'mic': recorded by the app, 'file': imported (how its confirmation is named)
# seconds  speech the profile was made from
# engines  {model id (models.toml, languages.clone_engine): {file, model}}: the voice of a profile without a recording
#          (kind 'them'): `file` in the folder (np.save of the engine's state or vector), `model` the sha256 of the
#          weights it was made with; other weights cannot use it (a contact is made again only from a new call).
#          'me' needs none: its engines cache what they need next to reference.wav and check it against the recording.
# sample   (them, an engine of the NVIDIA voice pack: its voice is an id in that call's runner, no state to keep) {file,
#          sha256, seconds}: up to THEM_S of their clean speech, `file` (16-bit WAV) in the folder; the next call's runner
#          makes their voice from it. Another file there (sha256) is not used: the voice is made again in the call
VERSION = 2
THEM_S = 10.0  # a contact's recording (`sample`): at most this long
CONSENT = {'mic-live': 'me', 'import+mic-match': 'me', 'call-saved': 'them'}  # how -> the kind it confirms


class NotMine(RuntimeError):
    """A profile that may not speak as me (into the call, to my server): a contact's, or mine not confirmed yet."""


def _sha(path: Path) -> str:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ''


def why_not_mine(v: Voice) -> str:
    """'' when the profile is my own voice confirmed at the microphone (M-20), else why not, for the user. The one
    check: the me side (virtual mic) and the server's voice go through `mine()`."""
    m = v.meta
    if m.get('kind', 'me') != 'me':
        return 'it is the voice of someone you talk to: it speaks only in your headphones, never into the call'
    c = m.get('consent') or {}
    if CONSENT.get(c.get('how')) != 'me':
        return 'it is not confirmed as your own voice yet: read two short phrases at the microphone (Voices…)'
    if not (sha := _sha(Path(v.path) / 'reference.wav')) or c.get('ref_sha256') != sha:
        return 'its recording changed after it was confirmed: confirm it again (Voices…)'
    return ''


def mine(name: str, root: Path | None = None) -> Voice:
    """The profile `name` if it may speak as me, else NotMine (the text names no voice: it may reach a log)."""
    v = load(name, root)
    if why := why_not_mine(v):
        raise NotMine(f'this voice cannot speak for you: {why}')
    return v


def upgrade(meta: dict, folder: str) -> dict:
    """A v1 voice.json as v2. v1 had only my voice, and nothing checked it (0.3.0's Record put the screen text in
    `transcript` whatever was heard): not confirmed until read at the microphone once (Voices…)."""
    if meta.get('version', 1) >= VERSION:
        return meta
    return meta | {'version': VERSION, 'id': folder, 'name': meta.get('name', folder), 'kind': 'me', 'consent': None,
                   'source': 'file', 'seconds': meta.get('quality', {}).get('seconds', 0.0), 'engines': {}}


def _private(*dirs: Path) -> None:
    for d in dirs:  # a voice (enough to clone it) and its states: the user's only (L-09; Linux, a shared machine with
        d.chmod(0o700)  # a 0755 home). The rest of the data folder keeps the umask


def _write(path: Path, meta: dict) -> None:
    import os
    import threading
    tmp = path / f'voice.json.{os.getpid()}-{threading.get_ident()}.tmp'  # both sides may upgrade a v1 profile at once
    try:
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, path / 'voice.json')
    finally:
        tmp.unlink(missing_ok=True)


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def save_them(name: str, lang: str, seconds: float, states: dict[str, tuple[np.ndarray, str]],
              root: Path | None = None, sample: tuple[np.ndarray, int] | None = None) -> Voice:
    """Keep a contact's voice from a call (an explicit "remember", the other person agreed): the engines' states, no
    recording of their speech; with the NVIDIA voice pack (no state to keep) `sample`, (pcm, sr) of their clean speech,
    written as a recording of at most THEM_S. `states`: {model id: (the engine's state or vector, sha256 of its weights)}."""
    import secrets
    import shutil

    import soundfile as sf

    from peresmeshnik import languages
    if gpu := [m for m in states if languages.needs_gpu(m)]:  # the voice pack's engines keep no voice (COMMON 23:49)
        raise ValueError(f'not kept: {", ".join(gpu)} keeps no voice for the next call')
    base = root or voices_dir()
    vid = f'them-{secrets.token_hex(4)}'
    tmp = base / f'{vid}.tmp'  # made whole aside, then renamed: never a folder without its voice.json
    tmp.mkdir(parents=True)
    _private(base, tmp)
    try:
        engines = {}
        for model, (state, sha) in states.items():
            np.save(tmp / f'{model}.npy', np.asarray(state), allow_pickle=False)
            engines[model] = {'file': f'{model}.npy', 'model': sha}
        meta = {'version': VERSION, 'id': vid, 'name': name, 'kind': 'them', 'lang': lang, 'created': _now(),
                'consent': {'how': 'call-saved', 'at': _now(), 'wer': None, 'similarity': None},
                'seconds': round(float(seconds), 2), 'engines': engines}
        if sample is not None:
            pcm, sr = np.asarray(sample[0], np.float32)[:int(THEM_S * sample[1])], sample[1]
            sf.write(tmp / 'sample.wav', np.clip(pcm, -1, 1), sr, subtype='PCM_16')
            meta['sample'] = {'file': 'sample.wav', 'sha256': _sha(tmp / 'sample.wav'), 'seconds': round(len(pcm) / sr, 2)}
        _write(tmp, meta)
        tmp.rename(base / vid)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return Voice(name=vid, engine='reference', path=base / vid, meta=meta)


APP_IDS = r'me|them-[0-9a-f]{8}'  # the app's own folder ids: they name nobody (crash reports keep them)
ASIDE = ('.new', '.old', '.tmp')  # the wizard's take in progress, its last one, a contact's voice being saved


def profiles(root: Path | None = None) -> list[Voice]:
    """The saved voices: mine and the contacts' (not the folders set aside, ASIDE; not the damaged ones)."""
    out = []
    for p in sorted((root or voices_dir()).glob('*/voice.json'), key=lambda p: p.parent.name):  # one order on every OS
        if not p.parent.name.endswith(ASIDE):
            try:
                out.append(load(p.parent.name, root))
            except (OSError, ValueError, KeyError, AttributeError):  # a broken voice.json: damaged()
                pass
    return out


def damaged(root: Path | None = None) -> list[str]:
    """Folders in the voices folder that are no voice: no or a broken voice.json (a crash mid-save); the voices
    window offers to delete them."""
    ok = {v.name for v in profiles(root)}
    return sorted(p.name for p in (root or voices_dir()).glob('*') if p.is_dir() and p.name not in ok
                  and not (p.name.endswith(('.new', '.old')) and (p / 'voice.json').is_file()))


def names(root: Path | None = None) -> set[str]:
    """What in the voices folder may name a person, for crash reports to scrub: the names on the voices screen and
    folders the user named (v1 `--name`); not the app's own ids ('me', 'them-<hex>'), which name nobody."""
    import re
    out = set()
    for p in (root or voices_dir()).glob('*'):
        out.add(p.name)
        try:
            out.add(json.loads((p / 'voice.json').read_text(encoding='utf-8')).get('name', ''))
        except (OSError, ValueError, AttributeError):
            pass
    out = {re.sub(r'\.(new|old|tmp)$', '', n) for n in out if n and isinstance(n, str)}
    return {n for n in out if not re.fullmatch(APP_IDS, n)}


def engine_state(v: Voice, model: str, sha: str) -> np.ndarray | None:
    """The state a profile keeps for the engine `model` with weights `sha`; None: not made with those weights."""
    e = v.meta.get('engines', {}).get(model)
    if not e or e.get('model') != sha:
        return None
    return np.load(v.path / Path(e['file']).name, allow_pickle=False)


def their_sample(v: Voice) -> tuple[np.ndarray, int] | None:
    """A contact's recording kept for the NVIDIA voice pack (save_them's `sample`): (pcm, sr); None without one, or when
    the file there is not the one kept."""
    s = v.meta.get('sample')
    if not s or not (sha := _sha(path := v.path / Path(s['file']).name)) or sha != s.get('sha256'):
        return None
    return read_audio(path)


# --- the live check: my voice is mine (M-20, RESEARCH-clone 6.1) ---

LIVE_PHRASES = {  # read at the microphone to confirm a voice, two at random (a recording made before cannot say them);
    # short everyday sentences written for the app, the same eight in every language
    'en': ['The weather is nice today, let us go for a walk.', 'I usually drink tea in the morning and coffee after lunch.',
           'Please close the window, it is getting cold in here.', 'My neighbour has a small dog that barks at every bird.',
           'We bought fresh bread and cheese at the market.', 'The train was late again, so I took a taxi home.',
           'She reads a book every evening before going to sleep.', "Could you send me the notes from yesterday's meeting?"],
    'ru': ['Сегодня хорошая погода, давай пойдём гулять.', 'Утром я обычно пью чай, а после обеда кофе.',
           'Закрой, пожалуйста, окно, здесь становится холодно.', 'У моего соседа маленькая собака, которая лает на птиц.',
           'Мы купили на рынке свежий хлеб и сыр.', 'Поезд опять опоздал, и я поехал домой на такси.',
           'Каждый вечер перед сном она читает книгу.', 'Пришлите мне, пожалуйста, заметки со вчерашней встречи.'],
    'uk': ['Сьогодні гарна погода, ходімо на прогулянку.', "Вранці я зазвичай п'ю чай, а після обіду каву.",
           'Зачиніть, будь ласка, вікно, тут стає холодно.', 'У мого сусіда є маленький пес, який гавкає на птахів.',
           'Ми купили на ринку свіжий хліб і сир.', 'Потяг знову запізнився, тож я поїхав додому таксі.',
           'Щовечора перед сном вона читає книжку.', 'Надішліть мені, будь ласка, нотатки з учорашньої зустрічі.'],
    'de': ['Heute ist schönes Wetter, lass uns spazieren gehen.', 'Morgens trinke ich meistens Tee und nach dem Mittagessen Kaffee.',
           'Mach bitte das Fenster zu, es wird hier kalt.', 'Mein Nachbar hat einen kleinen Hund, der jeden Vogel anbellt.',
           'Wir haben auf dem Markt frisches Brot und Käse gekauft.', 'Der Zug hatte wieder Verspätung, also nahm ich ein Taxi.',
           'Sie liest jeden Abend ein Buch, bevor sie schlafen geht.', 'Kannst du mir die Notizen von der gestrigen Besprechung schicken?'],
    'fr': ["Il fait beau aujourd'hui, allons nous promener.", 'Le matin, je bois souvent du thé et du café après le déjeuner.',
           "Ferme la fenêtre, s'il te plaît, il commence à faire froid.", 'Mon voisin a un petit chien qui aboie après chaque oiseau.',
           'Nous avons acheté du pain frais et du fromage au marché.', "Le train était encore en retard, alors j'ai pris un taxi.",
           'Elle lit un livre tous les soirs avant de dormir.', "Peux-tu m'envoyer les notes de la réunion d'hier ?"],
    'es': ['Hoy hace buen tiempo, vamos a dar un paseo.', 'Por la mañana suelo tomar té y después de comer café.',
           'Cierra la ventana, por favor, aquí empieza a hacer frío.', 'Mi vecino tiene un perro pequeño que le ladra a cada pájaro.',
           'Compramos pan fresco y queso en el mercado.', 'El tren volvió a llegar tarde, así que tomé un taxi a casa.',
           'Ella lee un libro todas las noches antes de dormir.', '¿Me puedes enviar las notas de la reunión de ayer?'],
    'it': ['Oggi è una bella giornata, andiamo a fare una passeggiata.', 'La mattina di solito bevo tè e dopo pranzo un caffè.',
           'Chiudi la finestra, per favore, qui comincia a fare freddo.', 'Il mio vicino ha un cagnolino che abbaia a ogni uccello.',
           'Abbiamo comprato pane fresco e formaggio al mercato.', 'Il treno era di nuovo in ritardo, così ho preso un taxi.',
           'Lei legge un libro ogni sera prima di dormire.', 'Mi puoi mandare gli appunti della riunione di ieri?'],
    'pt': ['Hoje está um dia bonito, vamos dar um passeio.', 'De manhã costumo tomar chá e depois do almoço café.',
           'Feche a janela, por favor, aqui está ficando frio.', 'O meu vizinho tem um cachorro pequeno que late para cada pássaro.',
           'Compramos pão fresco e queijo na feira.', 'O trem atrasou de novo, então peguei um táxi para casa.',
           'Ela lê um livro todas as noites antes de dormir.', 'Você pode me enviar as anotações da reunião de ontem?'],
    'pl': ['Dzisiaj jest ładna pogoda, chodźmy na spacer.', 'Rano zwykle piję herbatę, a po obiedzie kawę.',
           'Zamknij proszę okno, robi się tu zimno.', 'Mój sąsiad ma małego psa, który szczeka na każdego ptaka.',
           'Kupiliśmy na targu świeży chleb i ser.', 'Pociąg znowu się spóźnił, więc wróciłem do domu taksówką.',
           'Ona co wieczór czyta książkę przed snem.', 'Czy możesz mi wysłać notatki z wczorajszego spotkania?'],
    'tr': ['Bugün hava çok güzel, hadi yürüyüşe çıkalım.', 'Sabahları genellikle çay, öğleden sonra kahve içerim.',
           'Lütfen pencereyi kapat, burası soğumaya başladı.', 'Komşumun her kuşa havlayan küçük bir köpeği var.',
           'Pazardan taze ekmek ve peynir aldık.', 'Tren yine gecikti, bu yüzden eve taksiyle döndüm.',
           'Her akşam uyumadan önce kitap okur.', 'Dünkü toplantının notlarını bana gönderebilir misin?'],
    'zh': ['今天天气很好，我们出去散散步吧。', '我早上一般喝茶，午饭以后喝咖啡。', '请把窗户关上，这里有点冷了。',
           '我的邻居有一只小狗，看见鸟就叫。', '我们在市场上买了新鲜的面包和奶酪。', '火车又晚点了，所以我打车回家。',
           '她每天晚上睡觉前都看书。', '你能把昨天开会的笔记发给我吗？'],
    'ja': ['今日はいい天気なので、散歩に行きましょう。', '朝はたいていお茶を飲み、昼ご飯の後にコーヒーを飲みます。',
           '寒くなってきたので、窓を閉めてください。', '隣の人は、鳥を見るとほえる小さな犬を飼っています。',
           '市場で焼きたてのパンとチーズを買いました。', '電車がまた遅れたので、タクシーで帰りました。',
           '彼女は毎晩寝る前に本を読みます。', '昨日の会議のメモを送ってもらえますか？'],
    'ko': ['오늘은 날씨가 좋으니까 산책하러 가요.', '아침에는 보통 차를 마시고 점심 후에는 커피를 마셔요.',
           '추워지니까 창문 좀 닫아 주세요.', '우리 이웃집에는 새만 보면 짖는 작은 개가 있어요.',
           '시장에서 신선한 빵과 치즈를 샀어요.', '기차가 또 늦어서 택시를 타고 집에 갔어요.', '그녀는 매일 밤 자기 전에 책을 읽어요.',
           '어제 회의 메모를 저에게 보내 줄 수 있어요?'],
    'ar': ['الجو جميل اليوم، هيا نخرج في نزهة.', 'في الصباح أشرب الشاي عادة، وبعد الغداء أشرب القهوة.',
           'أغلق النافذة من فضلك، بدأ الجو يبرد هنا.', 'عند جاري كلب صغير ينبح على كل طائر.', 'اشترينا خبزا طازجا وجبنا من السوق.',
           'تأخر القطار مرة أخرى، فعدت إلى البيت بسيارة أجرة.', 'هي تقرأ كتابا كل مساء قبل النوم.',
           'هل يمكنك أن ترسل لي ملاحظات اجتماع الأمس؟'],
}
LIVE_WER = 0.3  # the live reading: the app's ASR heard the screen text (RESEARCH-clone 6.1, as loose as check_clone's)
SPEAKER = 'wespeaker-resnet34'  # registry id: the speaker vector ("the same voice"; S2: who is speaking now)
# cosine of the speaker vectors, the live reading (silence cut out) against the profile's recording. Stage 8, LibriSpeech
# dev-clean (40 people, another chapter as another session), 5 s live: another person of the same sex passes <= 0.64 %
# in every condition (17 423 pairs, p99 0.43); the same person is refused 2.8 % clean, 6-8 % with a laptop mic at
# SNR 20 dB, 14-20 % at 10 dB, 21 % against a phone-band file: a refusal asks for a quieter place, a try again helps
MATCH = 0.45
# the recording is checked window by window, each against the live reading (min): 30 s of someone else then 30 s of
# me must not pass as me, and the clone takes the recording's start (XTTS its first 30 s, Pocket its first 10 s).
# BENCH (FIXES-s3): such splices pass 88-90 % with one vector, 0 % with 8 s windows; the same person is refused 4.1 %
# clean (3.4 % with one vector), another person passes 0.06 % a try, 0.19 % in 3 tries. 10-12 s: fewer refusals in
# noise, more of another person in a mixed window
WIN_S = 8.0
# its start in shorter windows: a piece of someone else shorter than WIN_S at the very start (where the engines take their
# prompt: XTTS 6 s, Pocket 10 s) is diluted by my voice in an 8 s window. BENCH (stage 8 tails), the worst over a shift
# of 0-2 s: 3/4/5/6 s of another person passed 96/90/66/36 % with 8 s windows; with 4 s windows every 2 s over the first
# 12 s at 0.40 (a 4 s window of mine scores lower: p5 0.51 against 0.55 at 8 s; another person's p99 0.41 against
# 0.42) 36/23/3.5/0.9 %; the same person refused 4.1 % clean (3.8 %), another person passing a try 0.056 % (0.063 %)
HEAD_S, HEAD_WIN_S, HEAD_HOP_S, HEAD_MATCH = 12.0, 4.0, 2.0, 0.40
# what the engines take is the file's raw start, pauses and all (Pocket and Qwen3 its first 10 s, XTTS 6 s and its speaker
# vector 30 s): checked as it is, not through the relative trim of _speech (a voice of someone else turned down to just
# above the noise of my pauses fell under it whole, FIXES-s3-tails r2 1). What the check does not see the engines do not
# get either (r3 1): save() writes the frames of the start more than FLOOR_DB under the median speech frame as digital
# silence (the median: a few loud seconds do not move it), and the check takes all of the start but digital silence
# (a file put there by hand, quiet voice and all, is checked whole). BENCH (stage 8 tails, written as silence): 5 s of
# another person 3 dB above the noise of my pauses (SNR 30) in the file and passing 94 % with the trim, 1.1 % at 35 dB
# (17 % at 30: a lower floor cuts them into shreds the windows miss); the same person refused 4.41 % clean (4.08 %),
# another person 0.063 % (0.056 %); of my own speech p95 0.3 s goes silent
START_S, FLOOR_DB = 30.0, 35.0
# the start the engines take, against the rest of the same recording: a short piece of someone else there passes the live
# check in a 4 s window now and then (their first 3 s after 0-2 s of mine: 18-33 %), yet the same recording's own voice
# tells it at once (one session, one microphone). Each head window's cosine with the mean vector of the 8 s windows after
# it. BENCH (stage 8, before 0.3.0; `prompt8`, the file as save() writes it): the owner's files 0.58-0.69 at the lowest
# (p1 0.62-0.70; clean, Opus, noise in the file at SNR 30 / 20, 40 and 60 s), refused not once more at 0.60 (at 0.65: +1
# point clean); with the prompt below, their first 3 s pass 0-0.65 %, 4-6 s and quiet ones 0 %
HEAD_REST = 0.60
# with them one window more: the engines' own prompt as they hear it, the sound of the file's first PROMPT_S (all of it but
# digital silence: Pocket, Qwen3, Chatterbox take their voice from there, XTTS its first 6 s). A voice of someone else at
# the floor leaves 1-5 s there that the 4 s windows, filled up with my speech after it, do not hear, yet Qwen3 and
# Chatterbox speak from it (BENCH: the other person's 10 s at -57 dBFS, then me: cosine to them 0.44-0.56, to me 0.20).
# A prompt of less than PROMPT_MIN_S of sound is refused too: the clone would speak from a few frames of silence or of
# whoever is there (the owner's files: 6.5 s at the least, p1 6.9; another person's 10 s at -62 dBFS leaves 0.27 s, 98 %
# under 3 s; BENCH)
PROMPT_S, PROMPT_MIN_S = 10.0, 3.0
RECORD_AGAIN = 'record your voice again alone, with nobody else talking (setup wizard, "Your voice")'
TRIES, PAUSES_S = 3, (60, 300, 1500, 3600)  # failed checks in a row before a pause; the pauses after them, growing


def live_phrases(lang: str, n: int = 2) -> list[str]:
    """What the live check shows: `n` of the language's LIVE_PHRASES, a new pick each time."""
    import random
    return random.SystemRandom().sample(LIVE_PHRASES[lang], n)


def _to(pcm: np.ndarray, sr: int, to: int) -> np.ndarray:
    x = np.asarray(pcm, np.float32)
    x = x.mean(axis=1) if x.ndim > 1 else x
    return x if sr == to else np.interp(np.arange(0, len(x), sr / to), np.arange(len(x)), x).astype(np.float32)


def _under_floor(x: np.ndarray, sr: int) -> np.ndarray:
    """Which 30 ms frames of the first START_S of `x` save() writes as digital silence: more than FLOOR_DB under the
    median frame of the recording's speech (check()'s: above the noise of the pauses + 6 dB)."""
    f = _frames(x - x.mean(), sr)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    speech = db[db > max(float(np.percentile(live, 10)) + 6, -60.0)] if len(live) else live
    if not len(speech):
        return np.zeros(min(len(f), int(START_S * 1000 / 30)), bool)
    return db[:int(START_S * 1000 / 30)] < float(np.median(speech)) - FLOOR_DB


def _start(pcm: np.ndarray, sr: int) -> np.ndarray:
    """What is checked of a recording: its first START_S as the engines take it, only digital silence out (save() wrote
    what is under the floor as such), then the speech of what follows (_speech's)."""
    x = np.asarray(pcm, np.float32)
    f, n = _frames(x, sr), int(START_S * 1000 / 30)
    keep = np.mean(f[:n] ** 2, axis=1) > 1e-10  # not digital silence (-100 dB, of the samples as they are)
    rest = _speech(x[n * f.shape[1]:], sr) if len(f) > n else x[:0]  # a short one joins the last window
    return np.concatenate([(f[:n][keep] - x.mean()).reshape(-1), rest]).astype(np.float32)


def _windows(x: np.ndarray, sr: int) -> list[tuple[np.ndarray, float]]:
    """(piece, its threshold) of the speech `x`: its first HEAD_S in HEAD_WIN_S pieces every HEAD_HOP_S (HEAD_MATCH),
    the rest in WIN_S pieces (MATCH), a last one shorter than half of it joining the one before."""
    h, k, n = int(HEAD_S * sr), int(HEAD_WIN_S * sr), int(WIN_S * sr)
    m = min(h, len(x))
    at = list(range(0, max(m - k, 0) + 1, int(HEAD_HOP_S * sr)))
    if at[-1] + k < m:  # a start shorter than HEAD_S: its end is covered too
        at.append(m - k)
    head = [(x[i:i + k], HEAD_MATCH) for i in at]
    rest = [x[i:i + n] for i in range(h, len(x), n)]
    if len(rest) > 1 and len(rest[-1]) < n // 2:
        rest[-2:] = [np.concatenate(rest[-2:])]
    if rest and len(rest[-1]) < n // 2 and len(x) > h:  # a short tail after the start: the last start window takes it
        head[-1] = (np.concatenate([head[-1][0], rest.pop()]), HEAD_MATCH)
    return head + [(w, MATCH) for w in rest]


def _checks(voices: Path) -> Path:
    return voices.parent / 'voice-checks.json'  # beside the voices: a new or deleted profile does not reset it


def wait_s(root: Path | None = None) -> float:
    """Seconds before the next live check may run (0: now): after TRIES failures in a row, a growing pause
    (the threshold holds for one try: another person's best of many tries passes more often, BENCH)."""
    try:
        c = json.loads(_checks(root or voices_dir()).read_text(encoding='utf-8'))
        return max(0.0, float(c['until']) - time.time())
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def _tried(ok: bool, voices: Path) -> None:
    f = _checks(voices)
    try:
        fails = 0 if ok else int(json.loads(f.read_text(encoding='utf-8')).get('fails', 0)) + 1
    except (OSError, ValueError, AttributeError):
        fails = 0 if ok else 1
    pause = PAUSES_S[min(fails - TRIES, len(PAUSES_S) - 1)] if fails >= TRIES else 0
    f.write_text(json.dumps({'fails': fails, 'until': time.time() + pause}), encoding='utf-8')


def _speech(pcm: np.ndarray, sr: int, edges: bool = False) -> np.ndarray:
    """The speech of a recording: its 30 ms frames above the background, as check() counts them. `edges`: only the
    silence before the first and after the last such frame goes, the pauses stay (the live reading: in noise, cutting
    every quiet frame cuts quiet speech too, and the same person was refused 72 % at SNR 10 against 38 %, BENCH)."""
    x = np.asarray(pcm, np.float32)
    x = x.mean(axis=1) if x.ndim > 1 else x
    f = _frames(x - x.mean(), sr)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    on = db > max(float(np.percentile(live, 10)) + 6 if len(live) else -120.0, -60.0)
    if not on.any():
        return x
    i, j = np.flatnonzero(on)[[0, -1]]
    return (f[i:j + 1] if edges else f[on]).reshape(-1).astype(np.float32)


class Speaker:
    """A voice -> its speaker vector (unit length): wespeaker ResNet34 VoxCeleb on onnxruntime, run as onnx-asr 0.12.0
    (uv.lock) runs it: its numpy Kaldi fbank, mean over time removed, `feats` -> `embs`. No torch."""

    def __init__(self, model_dir: Path, threads: int = 1):
        import onnxruntime as rt
        from onnx_asr.preprocessors.numpy_preprocessor import KaldiPreprocessorNumpy

        from peresmeshnik.local.vad import ort_options
        self._fbank = KaldiPreprocessorNumpy('wespeaker')
        self._s = rt.InferenceSession(str(next(Path(model_dir).glob('*.onnx'))), ort_options(threads),
                                      providers=['CPUExecutionProvider'])

    def embed(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        x = _to(pcm, sr, 16_000)[None]
        f, _ = self._fbank(x, np.array([x.shape[1]], np.int64))
        e = self._s.run(['embs'], {'feats': (f - f.mean(axis=1, keepdims=True)).astype(np.float32)})[0][0]
        return e / (np.linalg.norm(e) + 1e-12)


def confirm_live(ref: Voice, pcm: np.ndarray, sr: int, said: list[str], asr, speaker: Speaker) -> tuple[dict | None, str]:
    """The live reading `pcm` of the shown `said` against my profile `ref`: (consent for voice.json, '') when the
    app's ASR (local/asr.OnnxAsr of the profile's language) heard the words and every window of the recording is the
    reader's voice; else (None, why, for the user). The consent names the recording it checked (ref_sha256): set_consent
    writes it only onto that one. After TRIES failures in a row it waits (wait_s)."""
    import io

    import soundfile as sf

    from peresmeshnik.engines import SAMPLE_RATE
    if w := wait_s(ref.path.parent):
        return None, f'Too many tries: wait {w / 60:.0f} min before the next one.'
    data = (ref.path / 'reference.wav').read_bytes()  # the bytes checked are the bytes the hash names
    x, rsr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    heard = asr.recognize(_to(pcm, sr, SAMPLE_RATE), lang=ref.meta.get('lang')).text
    err = wer(' '.join(said), heard)
    live = speaker.embed(_speech(pcm, sr, edges=True), sr)
    vecs = [(speaker.embed(w, rsr), m) for w, m in _windows(_start(x.mean(axis=1), rsr), rsr)]
    if len(p := _prompt(x.mean(axis=1), rsr)) >= rsr // 2:  # the engines' prompt, heard as a window of the head
        vecs.append((speaker.embed(p, rsr), HEAD_MATCH))
    sims = [(float(e @ live), m) for e, m in vecs]
    sim, worst = min(sims, key=lambda s: s[0] - s[1])  # the window furthest below its threshold (or nearest to it)
    low = sim < worst
    why = ([f'the words heard are not the ones on the screen ({err:.0%} differ): read them as written'] if err > LIVE_WER
           else []) + ([f'this is not the voice of the recording (similarity {sim:.2f})'] if low else [])
    if low and (r := check(pcm, sr)).speech_s and r.snr_db < FAIR_DB:  # the same person: refused 3x as often
        why.append(f'the background is loud (your speech only {r.snr_db:.0f} dB above it): try again somewhere quiet')
    if low and ref.meta.get('quality', {}).get('band_db', 0.0) < BAND_DB:
        why.append('the recording sounds like a phone call: one from a better microphone matches more reliably')
    _tried(not why, ref.path.parent)  # the recording's own start (below) is no try: another reading cannot change it
    if len(p) < PROMPT_MIN_S * rsr:
        why.append(f'the first {PROMPT_S:.0f} s of the recording, where the clone takes your voice from, are almost silent '
                   f'({len(p) / rsr:.1f} s of sound): start speaking right away, and {RECORD_AGAIN}')
    elif (head := head_rest(vecs)) < HEAD_REST:
        why.append(f'the start of the recording does not sound like the rest of it (similarity {head:.2f}): someone '
                   f'else may be speaking there; {RECORD_AGAIN}')
    if why:
        return None, 'Not confirmed: ' + '; '.join(why) + '.'
    import hashlib
    how = 'mic-live' if ref.meta.get('source') == 'mic' else 'import+mic-match'
    return {'how': how, 'at': _now(), 'wer': round(err, 3), 'similarity': round(min(s for s, _ in sims), 3), 'windows': len(sims),
            'ref_sha256': hashlib.sha256(data).hexdigest(), 'live': [round(float(e), 5) for e in live]}, ''


# --- the clone made from my recording speaks in the voice of my live reading (BRIEF-s3-clone-output-check) ---
# The end-to-end net behind the recording's checks above: before my voice is confirmed (where its engine can be made then),
# and else the first time an engine is made from it (vet_clone), the clone says the two test phrases of the language it
# speaks; one speaker vector of both against the live reading's (consent['live']). BENCH (stage 8, before 0.3.0: 88 own
# recordings x 3 live readings, Pocket / XTTS / Qwen3 / Chatterbox as the app runs them, phrases in en and, from an
# English reading, ru and de): the owner's lowest p1 0.205 (Qwen3 into Russian; XTTS 0.25, Pocket 0.32, Chatterbox
# 0.35), refused at 0.20: 0.8 % more there, 0.4 % into German, none elsewhere. It cannot tell a blend of 3-4 s of
# someone else from my own clone (their spread overlaps: those the recording's start refuses), but it refuses a clone
# that says nothing or nobody (Pocket on a silent start: 80 % of such files) and whatever an engine makes of a file the
# checks above passed
CLONE_MATCH = 0.20
CLONE_WHY = ("the voice made from your recording does not sound like you (similarity {sim:.2f}): someone else's speech "
             'may be in it; ' + RECORD_AGAIN)


def _speaker() -> Speaker:
    """The live check's speaker vector model (downloaded with it: languages.needed_models). A test replaces it."""
    from peresmeshnik import models
    return Speaker(models.path(SPEAKER))


def clone_match(tts, v: Voice, lang: str, live, speaker: Speaker | None = None) -> float:
    """The clone `tts` says the test phrases of `lang` in my profile `v`: the cosine of one speaker vector of both (0.3 s
    apart) with the live reading's `live` (consent['live']). Nothing said: 0."""
    gap, parts = np.zeros(int(0.3 * tts.sample_rate), np.float32), []
    for text in TEST_PHRASES.get(lang, TEST_PHRASES['en']):
        parts += [*(np.asarray(c, np.float32) for c in tts.synth(text, lang, v)), gap]
    pcm = np.concatenate(parts[:-1])
    if np.count_nonzero(pcm) < tts.sample_rate // 2:
        return 0.0
    return float((speaker or _speaker()).embed(pcm, tts.sample_rate) @ np.asarray(live, np.float32))


def _heard(engine: str, sim: float) -> dict:
    from peresmeshnik import models
    return {'model': models.get(engine).fingerprint, 'similarity': round(float(sim), 3)}


def confirm_clone(consent: dict, tts, engine: str, v: Voice, lang: str, speaker: Speaker | None = None) -> tuple[dict | None, str]:
    """After confirm_live passed: the clone `tts` (the registry's `engine`, made from the recording `v`, speaking `lang`)
    against that live reading: (the consent with the answer kept for `engine`, '') or (None, why, for the user). A
    refusal is a failed try (wait_s): the clone samples, another try must not be another draw."""
    sim = clone_match(tts, v, lang, consent['live'], speaker)
    if sim >= CLONE_MATCH:
        return consent | {'clones': {**consent.get('clones', {}), engine: _heard(engine, sim)}}, ''
    _tried(False, v.path.parent)
    return None, 'Not confirmed: ' + CLONE_WHY.format(sim=sim) + '.'


def clone_unlike(v: Voice, engine: str) -> str:
    """Why the clone `engine` may not speak my profile `v`, as already heard ('' = heard like me, or not heard yet with
    these weights: vet_clone hears it when it is made)."""
    from peresmeshnik import models
    k = ((v.meta.get('consent') or {}).get('clones') or {}).get(engine)
    if k and k.get('model') == models.get(engine).fingerprint and k['similarity'] < CLONE_MATCH:
        return CLONE_WHY.format(sim=k['similarity'])
    return ''


def vet_clone(tts, engine: str, v: Voice, lang: str) -> str:
    """'' when the clone `tts` (the registry's `engine`, made from my confirmed profile `v`) speaks in the voice of my
    live reading, else why, for the user. Heard once per recording and engine weights: the answer is kept in voice.json
    (consent['clones'], only onto the recording it heard), then read. The me side calls it after making the engine."""
    from peresmeshnik import models
    c = v.meta.get('consent') or {}
    k = (c.get('clones') or {}).get(engine)
    if k and k.get('model') == models.get(engine).fingerprint:
        return clone_unlike(v, engine)
    if not c.get('live'):  # confirmed by a build before this check: no live reading to hear the clone against
        return 'it was confirmed before the app checked the voice made from it: confirm it again (Voices…)'
    sim = clone_match(tts, v, lang, c['live'])
    try:
        now = load(v.name, v.path.parent)
        nc = now.meta.get('consent') or {}
        if nc.get('ref_sha256') == c.get('ref_sha256') == _sha(v.path / 'reference.wav'):
            _write(v.path, now.meta | {'consent': nc | {'clones': {**(nc.get('clones') or {}), engine: _heard(engine, sim)}}})
    except (OSError, ValueError):  # read-only, or gone meanwhile: heard again the next time
        pass
    return '' if sim >= CLONE_MATCH else CLONE_WHY.format(sim=sim)


def _prompt(x: np.ndarray, sr: int) -> np.ndarray:
    """What the engines take their voice from: the file's first PROMPT_S but digital silence (save() writes under its floor
    as such)."""
    f = _frames(np.asarray(x, np.float32)[:int(PROMPT_S * sr)], sr)
    return f[np.mean(f ** 2, axis=1) > 1e-10].reshape(-1).astype(np.float32)


def head_rest(vecs: list[tuple[np.ndarray, float]]) -> float:
    """confirm_live's window vectors (vector, threshold; the engines' prompt is one of the head): the lowest cosine of a
    head window (HEAD_MATCH) with the mean of the rest (MATCH; a recording too short to have any: of all of them)."""
    head = [e for e, m in vecs if m == HEAD_MATCH]
    r = np.mean([e for e, m in vecs if m != HEAD_MATCH] or [e for e, _ in vecs], axis=0)
    return min(float(e @ r) for e in head) / (float(np.linalg.norm(r)) + 1e-12) if head else 1.0


def set_consent(name: str, consent: dict | None, root: Path | None = None) -> Voice:
    """Write how my profile `name` was confirmed (confirm_live) into its voice.json: only onto the recording that
    was checked (a new take or import meanwhile: ValueError, nothing written)."""
    v = load(name, root)
    if consent and CONSENT.get(consent.get('how')) != v.meta['kind']:
        raise ValueError(f'{consent.get("how")!r} does not confirm a {v.meta["kind"]!r} voice')
    if consent and consent.get('ref_sha256') != _sha(v.path / 'reference.wav'):
        raise ValueError('the recording changed while it was checked: confirm it again')
    _write(v.path, v.meta | {'consent': consent})
    return load(name, root)


def save(pcm: np.ndarray, sr: int, name: str, lang: str, transcript: str | None = None,
         root: Path | None = None, source: str = 'file') -> Voice:
    """Write my voice `<root>/<name>/reference.wav` + `voice.json` (v2, kind 'me', not confirmed: confirm_live);
    returns the profile (engine 'reference')."""
    import soundfile as sf
    x = np.asarray(pcm, dtype=np.float32)
    x = (x.mean(axis=1) if x.ndim > 1 else x)[:int(MAX_S * sr)]
    report = check(x, sr)  # of what is kept: at most MAX_S
    peak = float(np.max(np.abs(x))) or 1.0
    gain = min(10 ** ((TARGET_DB - report.level_db) / 20), 10 ** (-1 / 20) / peak)
    base = root or voices_dir()
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    _private(base, path)
    for old in path.glob('*.pt'):  # engine caches of an older recording
        old.unlink()
    y = np.clip(x * gain, -1, 1)
    under = _under_floor(y, sr)
    fl = sr * 30 // 1000
    y[:len(under) * fl].reshape(-1, fl)[under] = 0  # the engines get only what is checked
    sf.write(path / 'reference.wav', y, sr, subtype='PCM_16')
    meta = {'version': VERSION, 'id': name, 'name': name, 'kind': 'me', 'lang': lang, 'created': _now(),
            'consent': None, 'source': source, 'seconds': report.seconds, 'engines': {},
            'transcript': transcript or '', 'sample_rate': sr, 'gain_db': round(20 * np.log10(gain), 1),
            'quality': {k: v for k, v in asdict(report).items() if k not in ('errors', 'warnings')}}
    _write(path, meta)
    return Voice(name=name, engine='reference', path=path, meta=meta)


def promote(draft: str, name: str, root: Path | None = None) -> Voice:
    """Make the profile `draft` the profile `name`, replacing it (the wizard keeps a new take aside until Finish)."""
    import shutil
    base = root or voices_dir()
    old = base / f'{name}.old'
    if old.exists():
        shutil.rmtree(old)
    if (base / name).exists():
        (base / name).rename(old)
    (base / draft).rename(base / name)
    if old.exists():
        shutil.rmtree(old)
    _write(base / name, upgrade(json.loads((base / name / 'voice.json').read_text(encoding='utf-8')), name)
           | {'id': name, 'name': name})
    return load(name, base)


def discard(name: str, root: Path | None = None) -> None:
    import shutil
    if (path := (root or voices_dir()) / name).exists():
        shutil.rmtree(path)


def load(name: str, root: Path | None = None) -> Voice:
    """A profile; a v1 voice.json is upgraded to v2 on disk once (an older app still reads it)."""
    path = (root or voices_dir()) / name
    meta = json.loads((path / 'voice.json').read_text(encoding='utf-8'))
    if not isinstance(meta, dict):  # JSON, but not a profile: as broken JSON (callers catch ValueError)
        raise ValueError('voice.json is not an object')
    if meta.get('version', 1) < VERSION:
        meta = upgrade(meta, name)
        try:
            _write(path, meta)
        except OSError:  # read-only: upgraded in memory again next time
            pass
    return Voice(name=name, engine='reference', path=path, meta=meta)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return x.mean(axis=1), sr


def _print_report(r: Report) -> None:
    print(summary(r))
    print(verdict(r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m peresmeshnik.voice', description='Record or import your own voice (30-60 s).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    en = sub.add_parser('enroll', help='make a voice profile from a recording')
    src = en.add_mutually_exclusive_group(required=True)
    src.add_argument('--import', dest='file', type=Path, help='an existing recording (wav, flac, ogg, mp3)')
    src.add_argument('--record', type=float, nargs='?', const=45.0, metavar='SECONDS', help='record from the microphone (default 45 s)')
    en.add_argument('--name', default='me')
    en.add_argument('--lang', default='ru', help='language spoken in the recording')
    en.add_argument('--transcript', help='what is said (text or @file); --record uses the prompt text')
    en.add_argument('--device', help='microphone: Linux: a PipeWire source name (pactl list short sources); macOS: a '
                                     'Core Audio UID or name (peresmeshnik-audio list); elsewhere a PortAudio name or index')
    en.add_argument('--force', action='store_true', help='keep the profile even if the checks fail')
    en.add_argument('--engine', choices=['xtts'], help='also prepare the clone engine now (needs the voice pack)')
    en.add_argument('--model-dir', type=Path, help='XTTS-v2 model folder (config.json, model.pth, vocab.json)')
    ck = sub.add_parser('check', help='check a recording without saving')
    ck.add_argument('file', type=Path)
    sub.add_parser('list', help='list voice profiles')
    for p in (en, sub.choices['list']):
        p.add_argument('--voices', type=Path, help=f'profiles folder (default {voices_dir()})')
    a = ap.parse_args(argv)
    if getattr(a, 'engine', None) and not a.model_dir:
        ap.error('--engine needs --model-dir')

    if a.cmd == 'list':
        for v in profiles(a.voices):
            m = v.meta
            print(f"{m['name']}\t{m['lang']}\t{m['seconds']:.0f} s\t{m['created']}\t{m['kind']}\t"
                  f"{(m['consent'] or {}).get('how', 'not confirmed')}\t{v.path}")
        return 0
    if a.cmd == 'check':
        r = check(*read_audio(a.file))
        _print_report(r)
        return 0 if r.ok else 1

    transcript = a.transcript
    if transcript and transcript.startswith('@'):
        transcript = Path(transcript[1:]).read_text(encoding='utf-8').strip()
    if a.file:
        pcm, sr = read_audio(a.file)
    else:
        from peresmeshnik.audio.mic import record
        if not transcript and a.lang not in PROMPTS:
            ap.error(f'no prompt text in {a.lang!r} ({", ".join(PROMPTS)}): pass --transcript with what you will read')
        transcript = transcript or PROMPTS[a.lang]
        print(f'Read this aloud at your normal pace, in a quiet room ({a.record:.0f} s):\n\n{transcript}\n')
        for i in (3, 2, 1):
            print(f'{i}...', end=' ', flush=True)
            time.sleep(1)
        print('recording', flush=True)
        sr = 24_000  # above the 16 kHz pipeline rate: the clone engine uses 22.05 kHz
        device = int(a.device) if a.device and a.device.isdigit() else a.device
        pcm = record(a.record, device=device, sample_rate=sr)
    r, v = enroll(pcm, sr, a.name, a.lang, transcript, a.force, a.voices, 'file' if a.file else 'mic')
    _print_report(r)
    if v is None:
        print('Not saved (keep it anyway with --force).')
        return 1
    print(f'saved voice profile {v.name!r}: {v.path}')
    if a.engine == 'xtts':
        from peresmeshnik.local.clone import make_clone
        make_clone(a.model_dir).latents(v)
        print('XTTS-v2 conditioning ready')
    return 0


if __name__ == '__main__':
    sys.exit(main())
