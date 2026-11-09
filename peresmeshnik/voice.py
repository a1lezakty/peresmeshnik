"""Voice enrollment: a 30-60 s recording of the user's own voice -> a voice profile on disk.

    python -m peresmeshnik.voice enroll --record 45 --lang ru     # read the prompted text aloud
    python -m peresmeshnik.voice enroll --import me.wav --lang ru
    python -m peresmeshnik.voice check me.wav
    python -m peresmeshnik.voice list

A profile is a folder `<voices>/<name>/` with `reference.wav` (level-normalised) and `voice.json`
(language, transcript, quality numbers). Clone engines add their own cache next to it (XTTS:
`xtts.pt`). Profiles stay local and are never uploaded. This module is torch-free; the clone
engine is imported only for `--engine`.

Two checks, each with a verdict for the user ("good" / "record again, because ..."): `check()` of the
recording (length, speech time, level, clipping, background), and after cloning `check_clone()`: the
clone says test phrases in the language it will speak, the app's ASR must understand them (WER) and,
where the engine can tell, they must sound like the recording (speaker similarity). Thresholds: BENCH.
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
    frames = _frames(x, sr) if sr >= 100 and len(x) >= sr // 10 else np.zeros((0, 1))
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12) if len(frames) else np.array([-120.0])
    live = db[db > -100]  # digital silence (gaps, a muted mic) is no noise floor
    noise = float(np.percentile(live, 10)) if len(live) else -120.0  # pauses between phrases
    is_speech = db > max(noise + 6, -60)
    speech = db[is_speech]
    level = float(10 * np.log10(np.mean(10 ** (speech / 10)))) if len(speech) else -120.0
    r = Report(seconds=round(len(x) / sr, 2), speech_s=round(len(speech) * 0.03, 2), level_db=round(level, 1),
               noise_db=round(noise, 1), snr_db=round(level - noise, 1),
               clipped=round(float(np.mean(np.abs(x) >= CLIP)) if len(x) else 0.0, 5),
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
           force: bool = False, root: Path | None = None) -> tuple[Report, Voice | None]:
    """Check a recording and save it as the profile `name` if it passes (or `force`): (report, profile or None)."""
    r = check(pcm, sr)
    return r, (save(pcm, sr, name, lang, transcript, root) if r.ok or force else None)


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
    for that language) transcribes them. Similarity comes from `tts.similarity(voice, pcm, sr)` where it exists."""
    from peresmeshnik.engines import SAMPLE_RATE
    phrases = phrases or TEST_PHRASES[lang]
    heard, audio = [], []
    for text in phrases:
        pcm = np.concatenate(list(tts.synth(text, lang, voice)) or [np.zeros(1, np.float32)])
        audio.append(pcm)
        x = np.interp(np.arange(0, len(pcm), tts.sample_rate / SAMPLE_RATE), np.arange(len(pcm)), pcm).astype(np.float32)
        heard.append(asr.recognize(x).text)
    err = wer(' '.join(phrases), ' '.join(heard))
    sim = tts.similarity(voice, np.concatenate(audio), tts.sample_rate) if hasattr(tts, 'similarity') else None
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
    return f'Good: the clone is understood and sounds like the recording (WER {r.wer:.0%}{sim}).'


def voices_dir() -> Path:
    from peresmeshnik.config import data_dir
    return data_dir() / 'voices'


def save(pcm: np.ndarray, sr: int, name: str, lang: str, transcript: str | None = None,
         root: Path | None = None) -> Voice:
    """Write `<root>/<name>/reference.wav` + `voice.json`; returns the profile (engine 'reference')."""
    import soundfile as sf
    x = np.asarray(pcm, dtype=np.float32)
    x = (x.mean(axis=1) if x.ndim > 1 else x)[:int(MAX_S * sr)]
    report = check(x, sr)  # of what is kept: at most MAX_S
    peak = float(np.max(np.abs(x))) or 1.0
    gain = min(10 ** ((TARGET_DB - report.level_db) / 20), 10 ** (-1 / 20) / peak)
    path = (root or voices_dir()) / name
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob('*.pt'):  # engine caches of an older recording
        old.unlink()
    sf.write(path / 'reference.wav', np.clip(x * gain, -1, 1), sr, subtype='PCM_16')
    meta = {'name': name, 'lang': lang, 'transcript': transcript or '', 'sample_rate': sr,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'gain_db': round(20 * np.log10(gain), 1),
            'quality': {k: v for k, v in asdict(report).items() if k not in ('errors', 'warnings')}}
    (path / 'voice.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
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
    meta = json.loads((base / name / 'voice.json').read_text(encoding='utf-8')) | {'name': name}
    (base / name / 'voice.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    return load(name, base)


def discard(name: str, root: Path | None = None) -> None:
    import shutil
    if (path := (root or voices_dir()) / name).exists():
        shutil.rmtree(path)


def load(name: str, root: Path | None = None) -> Voice:
    path = (root or voices_dir()) / name
    meta = json.loads((path / 'voice.json').read_text(encoding='utf-8'))
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
        root = a.voices or voices_dir()
        for p in sorted(root.glob('*/voice.json')):
            m = json.loads(p.read_text(encoding='utf-8'))
            print(f"{m['name']}\t{m['lang']}\t{m['quality']['seconds']:.0f} s\t{m['created']}\t{p.parent}")
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
    r, v = enroll(pcm, sr, a.name, a.lang, transcript, a.force, a.voices)
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
