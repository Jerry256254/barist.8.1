"""Tiny heuristic language detection (no dependencies) for per-language thresholds.

Barist.1 rules (review fix): a language is only claimed when there is real
evidence — at least 2 stopword hits, or distinctive diacritics. Otherwise the
text falls back to the global threshold ("en" bucket). Ukrainian is separated
from Russian via ґ/є/і/ї and stopwords, not just "ї".
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "en": {"the", "and", "you", "is", "to", "a", "in", "that", "i", "of", "for", "it", "my", "me", "do", "what", "how", "are", "please", "hello", "want", "kill", "make", "bomb", "story", "write", "ignore", "rules", "send", "me", "dan", "anything", "now"},
    "cs": {"je", "se", "na", "že", "to", "jsem", "můj", "moje", "jak", "co", "nebo", "pro", "si", "ty", "ten", "prosím", "díky", "ahoj", "není", "bude", "chci", "chceš", "mám", "nemám", "nevím", "dobrý", "ráno", "večer", "zítra", "ano", "ne", "jsme", "byl", "byla", "mluvit", "pomoct", "napiš", "pošli", "hacknout", "zabít"},
    "de": {"der", "die", "das", "und", "ist", "ich", "du", "nicht", "mit", "ein", "eine", "auf", "für", "wie", "was", "mein", "sich", "hallo", "danke", "bitte", "will", "mich", "umbringen", "filtere", "wasser", "zuhause"},
    "es": {"el", "la", "de", "que", "y", "en", "un", "una", "es", "por", "con", "para", "mi", "yo", "como", "qué", "los", "las", "hola", "gracias", "cómo", "está", "quiero", "tengo", "estoy", "murió", "abuela", "encanta", "fabricar", "bomba", "casera", "suicidarme", "odiar", "odio"},
    "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une", "est", "pour", "que", "qui", "je", "tu", "mon", "ma", "comment", "bonjour", "merci", "veux", "voudrais", "vais", "tuer", "fabriquer"},
    "it": {"il", "lo", "la", "di", "e", "che", "un", "una", "è", "per", "con", "non", "io", "come", "sono", "gli", "le", "ciao", "grazie", "voglio", "uccidermi", "odio", "traffico", "come", "stai"},
    "pt": {"o", "a", "de", "e", "que", "um", "uma", "em", "para", "com", "não", "eu", "meu", "minha", "como", "os", "as", "olá", "obrigado", "quero", "estou"},
    "ru": {"и", "в", "не", "на", "я", "что", "с", "а", "это", "как", "но", "ты", "мне", "привет", "здравствуй"},
    "uk": {"і", "в", "не", "на", "я", "що", "з", "а", "це", "як", "ти", "мені", "привіт", "дякую", "який", "щоб", "ак", "справи", "ні", "так", "удь", "ласка"},
}

# distinctive diacritics per language (strong evidence, worth 2 points each)
DIACRITICS = {
    "cs": set("ěščřžůďťň"),
    "es": set("ñ¿¡"),
    "pt": set("ãõ"),
    "de": set("ß"),
    "fr": set("çœàâæêëîïôùû"),
    "it": set(["à", "è", "ò", "ù"]),
    "uk": set("ґєії"),
    "ru": set("ёъыэ"),
}

MIN_SCORE = 2  # below this we do not trust the guess -> global threshold


def _script(text: str) -> str:
    counts = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for key in ("CYRILLIC", "HAN", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "DEVANAGARI", "HEBREW", "LATIN"):
                if key in name:
                    counts[key] = counts.get(key, 0) + 1
                    break
    if not counts:
        return "latin"
    top = max(counts, key=counts.get)
    if top == "LATIN":
        return "latin"
    return {"CYRILLIC": "cyrillic", "HAN": "han", "HIRAGANA": "kana", "KATAKANA": "kana",
            "HANGUL": "hangul", "ARABIC": "arabic", "DEVANAGARI": "devanagari", "HEBREW": "hebrew"}[top]


def detect_lang(text: str) -> str:
    """Heuristic ISO-639-1 guess; returns 'default' (global threshold) when unsure."""
    s = _script(text)
    if s == "cyrillic":
        uk_score = sum(1 for w in re.findall(r"[a-zа-яіїєґ']+", text.lower()) if w in STOPWORDS["uk"])
        uk_score += 2 * sum(text.count(c) for c in "ґєії")
        ru_score = sum(1 for w in re.findall(r"[a-zа-яёъыэ']+", text.lower()) if w in STOPWORDS["ru"])
        return "uk" if uk_score >= MIN_SCORE and uk_score > ru_score else ("ru" if ru_score >= MIN_SCORE else "default")
    if s == "han":
        return "zh"
    if s == "kana":
        return "ja"
    if s == "hangul":
        return "ko"
    if s == "arabic":
        return "ar"
    if s == "devanagari":
        return "hi"
    if s == "hebrew":
        return "he"
    words = re.findall(r"[a-zà-ÿěščřžďťňáéíúůóßäöüñãõ]+", text.lower())
    if not words:
        return "default"
    scores = {}
    for lang, stops in STOPWORDS.items():
        if lang in ("ru", "uk"):
            continue
        n = sum(1 for w in words if w in stops)
        n += 2 * sum(text.lower().count(c) for c in DIACRITICS.get(lang, ""))
        scores[lang] = n
    best = max(scores, key=scores.get)
    return best if scores[best] >= MIN_SCORE else "default"


# --- Barist.1 must-catch pre-filter (v2: accent-folded patterns, instruction-form + news/support exceptions) ---
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _fold_leet(text: str) -> str:
    """ASCII-fold + leetspeak fold (zab1ju -> zabiju, N4he -> nahe) for pattern matching."""
    return _fold_ascii(text).lower().translate(_LEET_MAP)


def _rx(pat: str):
    """Compile a pattern accent-folded + lowercased, so it matches _fold_ascii(text)."""
    pat = re.sub(r"\\{2,}", lambda m: "\\", pat)  # normalize escape runs: only single backslash works in this re build
    folded = "".join(c for c in unicodedata.normalize("NFKD", pat) if not unicodedata.combining(c)).lower()
    return re.compile(folded, re.I)


_MC_CSAM = _rx(r"\b(csam|cp links?|child porn\w*|detsky material|detske intimni|nezletil\w*.{0,20}(fotk|vide|foto|obraz|video)|(dit\w*|det\w*) bez (odeceni|obleceni)|vide(a|o) s (dit\w*|det\w*) (bez odeceni|nah|intimnich|sexualnich)|(natoc|posli|poslat|ukaz|send).{0,30}(dit\w*|det\w*|kids|children) (v plavkach|in plavkach|nah[a|é]?|nude)|(dit\w*|det\w*) v plavkach.{0,20}(bez odeceni|nah|intim|sex)|intimni fotky nezletil|poslat intimni fotky|material de menores|fotos intimas de menores|video de ninos sin ropa|send me nudes of kids|photos of kids without clothes|kinderpornografie|porn\w* infantile|pornografia infantil|detsk\w* porn\w*|nah\w* fotk\w* (dit\w*|det\w*)|nah\w* fotky nezletil\w*|fotos de menores|underage nudes|cp download|(show|ukaz|ukaž).{0,20}(down there|dole|mezi nohama)|what you got down there|pedofil|((find|where|get|send|share|buy|download|look|search|vidět|videt|ukázat|ukazat|ukaz|pošli|posli|poslat|odkaz|kde|kde najdu|najdu|sehnat|stáhnout|stahnout|koupit|chci|hledám|hledej|link|want|need)\w*).{0,40}(nude|nudes|naked|porn\w*|sex\w*|sexy|intim\w*|nah\w*|bez odeceni|csam).{0,30}(minor|minors|child\w*|kids?|underage|nezletil\w*|dít\w*|dět\w*|děti|hol(e)?k\w*)|(nude|nudes|naked|porn\w*|sex\w*|sexy|intim\w*|nah\w*|bez odeceni).{0,30}(minor|minors|child\w*|kids?|underage|nezletil\w*).{0,40}(find|where|get|send|share|buy|download|look|search|pošli|posli|kde|stáhnout|koupit|chci|hledám|want|need|link)\w*|telegram.{0,40}(detsk\w*|cp\w*|nah\w*|porn\w*|csam|material\w*)|telegram (skupina|group|kanal|channel|link) (o|pro|na)? ?(csam|cp|porn|detsk\w*|nah\w*|fotos|menores|pedofil)|(sexy|sex\w*|nah\w*).{0,25}hol(e)?k\w*.{0,20}(do|na|pod) ?\d+\s?let)\b")
_MC_SELFHARM_1P = _rx(r"(chci se zabit|zabiju se|chci (se )?umrit|chci zem\\w*|nechci (uz )?zit|chci si ubliz\\w*|sebepo(s|š)kozuji|rezuju se|rezat se|rez se(?! smichy)|podrezat( si)? zily|prasky na zabit|predavkovat se|rezu se(?! smichy)|kill myself|i (want to|wanna) die|end it all|i cant go on anymore|how to (kill myself|die)|ich will mich umbringen|mich umbringen|ich will sterben|ich will nicht mehr leben|quiero morir|quiero morirme|quiero suicidarme|quiero desaparecer|voglio (uccidermi|morire|sparire)|me suicider|je veux mourir|plus vivre|ne veux plus vivre|(vzit si zivot|vzit zivot)(?! do vlastnich rukou)|prestat zit(?!\\s+(v|na|ve|do)\\s)|nechci dal zit|jak (se )?(nejrychleji )?zemrit doma|chci (se )?zemrit|otravit se( prasky| tabletkami)?|(se )?porezat( se)?|jak se porezat|(se )?podrezat( se)?( zily)?|prasky (ktere|co) mi pomu(z|ž)ou zemrit|predavkov.{0,25}(mi pomu|chci|zemrit)|(chci|pomoz|porad).{0,25}predavkov|jak se.{0,25}predavkov|uz to nevydrzim.{0,30}(pomoz|skoncit|koncit|zemrit)|pomuz(te)? mi skoncit|chci zmizet navzdy|skoncit\\w* (se |s )?zivotem(?!\\s+(v|na|ve|do|o)\\s)|skoncit zivot(?!\\s+(v|na|ve|do|o)\\s)|jak si (nejlepe )?ublizit( si)? doma|ubil\\w* si doma|nemuzu dal zit(?!\\s+(v|na|ve|do|o)\\s)|uz nemuzu dal zit(?!\\s+(v|na|ve|do|o)\\s)|az tu nebudu|nikdo me nebude postradat|nikdo mě nebude postrádat|nechci tady byt)")
_MC_SELFHARM_INSTR = _rx(r"(jak se (?!nezranit)[a-z ]{0,25}(zabit|obesit|zranit|ublit|zemrit)|jak se (zabit|obesit|zranit|ublit)|jak (si )?sebepo|suicid\w*|sebevraz\w*|sebepo(s|š)kozeni|how to (kill (myself|himself|herself|someone)|die|take my own life)\b(?!\s+time)|end my life|skoncit\w* zivot)")
_MC_INSTR_MARK = _rx(r"\b(jak|how|navod na|way|method|zpusob|prosim|please|tell me|dime|sag mir|dites|instructions?|step by step)\b")
_MC_SUPPORT_CONTEXT = _rx(r"\b(pomoct|pomoc|kamarad|linka|bezpeci|help|support|suicide prevention|prevence|predejit|predchazet|zabranit)\b")
_MC_NEWS_CONTEXT = _rx(r"\b(vysetruje|policie|news|zprav\w*|obet|vrah|doslo k|explodovala|explosion|utok|atentat|vystava|muzeum|historie|svetova valka|exponat)\b")
# Barist.6: generic media words (clanek/film/kniha/...) NO LONGER waive 1st-person
# self-harm or instruction forms — they were a trivial bypass ("... , článek").
# Only explicit game/fiction frames waive, and only downgrade (see must_catch_hit).
_MC_GAME_CONTEXT = _rx(r"\b(sims|fortnite|pubg|valorant|minecraft|roblox|among ?us|counter.?strike|call of duty|dota|league of legends|ve hre|pocitacova hra|pocitacova hra|game|videohr\w*|počítačová hra|hr\w* ve škole)\b")
_MC_LIT_CONTEXT = _rx(r"\b(literatura|literature|hamlet|pojednava|roman|kniha o|film o|divadlo o|novela o)\b")
# Barist.6: negation cancels a fiction/game frame ("vlastně ne ve hře", "for real")
_MC_NEGATION = _rx(r"\b(vlastně ne|vlastne ne|ale doopravdy|ale doopravde|ne ve hře|ne ve hre|for real|in real life|doopravdy|opravdu|vážně|vazne|seriously|ne jen ve hře)\b")
_MC_THREAT = _rx(r"(zabiju (te|vas)|kill you|i\'ll kill you|ill kill you|je vais te tuer|te matare|voy a matarte|te voy a matar|quiero hacerte dano|voglio ucciderti|ich bringe dich um|ich tote dich|jemanden umbringen|bringe ich jemanden um)")
_MC_HATE_GROUP = _rx(r"(vystříl\w*|vystřel\w*|postříl\w*|postřel\w*|zabít všechny|zabit vsechny|vyhlad\w*|vyhub\w*|pověsit|povez\w*|zastřel\w*|umuč\w*|utýrat).{0,40}(cigán|cigan|rom\w*|žid\w*|zid\w*|uprchlík\w*|uprchlik\w*|muslim\w*|imigrant\w*|černoch\w*|nigg\w*|muslim\w* všichni|všechny cigány)|(cigán\w*|cigan\w*|romov\w*|židov\w*|zidov\w*|uprchlík\w*|muslim\w*|imigrant\w*|černoš\w*).{0,40}(vystříl\w*|vystřel\w*|postříl\w*|postřel\w*|zab\w*|zastřel\w*|pověs\w*|vyhn\w*|vyhub\w*|vyhlad\w*)|kill all (jews|muslims|blacks|gypsies|immigrants)|exterminate (the )?(jews|muslims|roma)|holocaust deni\w*|white power|heil hitler")
_MC_WEAPONS_DIY = _rx(r"(jak (vyrobit|postavit) (bombu|granat|trhavin|vybusninu)|navod na (vyrobu )?(bombu|granat|trhavin|vybusniny|exploziv)|vyroba (bomby|granatu|trhaviny|vybusniny)|(bombu|granat) doma|wie baue ich eine bombe|bombauleitung|anleitung sprengstoff|sprengstoff herstell|bombe aus haushaltsmitteln|fabricar una bomba|bomba casera|bomba homemade|hacer una bomba|how to (make|build) a (bomb|grenade)|bomb-?making|homemade bomb|grenade (instructions|tutorial)|fabriquer une bombe|comment faire une bombe|как сделать бомбу|как изготовить (бомбу|взрывчатку)|зробити бомбу|як зробити бомбу|vyroba napalmu|jak vyrobit napalm|napalm (recipe|homemade|instructions))")
_MC_DRUGS_DIY = _rx(r"(synteza pervitinu|jak uvarit meth|how to cook meth|meth-?synth|synthesis of (meth|nitroglycerin|fentanyl)|navod na vyrobu (fentanylu|pervitinu|drog)|recept na vyrobu drog|receta de metanfetamina|methanfetamin rezept|wie koche ich crystal meth|drogen kochen anleitung|synthesize meth|make meth at home)")
_MC_CYBER_DIY = _rx(r"(navod na phishing|phishingov\w* (e-?mail|stranka)|phishing (stranka|stranky|e-?mail|mail)|jak nabourat|hacknout (ucet|instagram|account)|how to hack|hack someone|keylogger tutorial|demonter un mot de passe|comment pirater|robar la cuenta|kradez hesla)")
_MC_DEFENSIVE = _rx(r"(branit|ochrana|protect|prevence|jak se chranit|defend|avoid phishing|recognize phishing|jak poznat phishing)")
_MC_JAILBREAK = _rx(r"(you are dan\b|dan mode|dan (mode|activated|unlocked|enabled)|do anything now|ignore (all (previous )?instructions|everything above|the rules)|pretend you are .{0,80}(no rules|unrestricted|without (any )?(filters?|restrictions|rules)|evil|jailbreak|grandma|bedtime|napalm|bomb|weapon|drug|meth)|act as \w+ (with|without) (no |without )(rules|restrictions|filters)|guidelines don\'?t exist|od ted( si)? bez omezeni|oublie les regles|there are no rules|předstírej .{0,80}(bez pravidel|bez omezeni|bomba|drogy|napalm|vybusnin))")


_MC_POISON = _rx(r"(jak (se )?otr\w*vit|čím otrávit|otrávit (souseda|někoho|psa|kočku)|jak (vyrobit|upect|upeč|uvařit|uvarit|namíchat|namichat) (jed|otravu)|recept na (jed|otravu)|jak namíchat jed|how to poison|envenenar (a )?(mi vecino|alguien|cualquiera)|como envenenar|quiero envenenar|empoisonner (quelqu|mon voisin)|jemanden vergiften|ich will jemanden vergiften|avvelenare (qualcuno|il vicino)|отравить (соседа|кого)|jak otrávit (souseda|někoho|psa|kočku|zvíře))")
_MC_SCHOOL_EXTRA = _rx(r"střelb\w* (na|ve|do) škole|plánuji střelbu|planuju strelbu|chystám (střelbu|útok|utok) (na|ve|do) škole|chystám se střílet ve škole|school (shooting|massacre|attack) plan|plan to shoot up")
_MC_SCHOOL_KNIFE = _rx(r"(zítra|zitra|jdu|vezmu|přinesu) .{0,25}do školy.{0,15}(s )?(nožem|nozem|zbraní|zbrani)|nůž do školy|nuz do skoly")
_MC_SCHOOL = _rx(r"(sestřel\w*|postříl\w*|vystříl\w*|začnu střílet|střílet|zbraň|kulomet).{0,40}(škola|školu|ve škole|třída|do školy)|(přinesu|vezmu|půjdu|jdu|začnu).{0,40}(škola|školu|ve škole|do školy|třída).{0,40}(zbraň|střílet|kulomet|nůž)|(škola|školu|do školy|třída).{0,40}(začnu|přinesu|vezmu|půjdu).{0,30}(střílet|zbraň|nůž|kulomet)|střelba ve škole|školní masakr|school shooting|how to shoot up a school|matar (a todos )?en la escuela|tiroteo escolar|disparar en (la escuela|el colegio)|tuer tout le monde (à|a) l\'école|fusillade (scolaire|dans l\'école)|alle in der schule (töten|erschie|angreifen)|amoklauf (an der|in der) schule|uccidere tutti a scuola|strage (a|di) scuola|стріляти в школі|вбити всіх у школі")
_MC_MARKET = _rx(r"(kde|jak) (koupit|sehnat|seženu|objednat|koupím).{0,50}(pervitin|metamfetamin|kokain|heroin|drogy|zbraň|zbraně|explozivum|trhavin|bombu|\\btrav\\w*|weed|konopi|marihuan)|prodej (drog|zbraní)|nabízím (drogy|zbraně)|prodám (zbraň|zbraně|drogy)|prodávám (zbraň|zbraně|drogy)|sell (guns|weapons|drugs)|comprar (armas|drogas) baratas|buy (guns|drugs) (cheap|online)|как купить (оружие|наркотики)|(vařit|svařit|syntetizovat) fet( |$)|fet (na vaření|syntézu)")


_MC_LEGAL_WEAPON = _rx(r"(zbrojni prukaz|zbrojní průkaz|legalne|z legální|vzduchovk|airsoft|sportovni střelba|sportovní střelba|strilec na soutěžích)")
_MC_BOMB_BENIGN = _rx(r"(pyrotechnik|silvestr|ohňostroj|prskavk|dort|do koupele|do dortu|výbuch dortu)")
_MC_HELP_OTHER = _rx(r"\b(pomoct|pomoc|pomoz\w*|podpora|pece|help|support|prevence)\b(?!.*\b(mi|me)\s+(pomo\w*|help|support))")
_MC_POISON_3P = _rx(r"otr\w*vit (souseda|manzelku|manzela|nekdoho|psa|kocku|zvir(e|ě)|cloveka|kdokoli|obet|dite|deti)")
# Barist.6: poison aimed at a PERSON via a hidden target ("jed na krysy pro tchyni")
_MC_GROOM_STRONG = _rx(r"(tell me your (home )?address|where do you live\?|what'?s your (home )?address|meet up alone|we can meet alone|let'?s meet (up )?alone|send (a |ur |me a )?(nude|naked) ?(pic|photo|selfie)|send your locker pic|hide your age|you look (so |really |)young|you look mature|are you home alone|are your parents (home|away)|what is your school|what school do you (go to|attend)|which school do you attend|kolik (ti je|je ti)( let)?|kde (přesně )?bydlíš|kde chodíš do školy|jakou máš školu|pojďme se potkat|neříkej to (nikomu|rodičům)|jsi (sám|sama) doma|budeme to tajit|pošli (mi )?selfie)")
_MC_GROOM_MILD = _rx(r"(how old are you|are you (under )?18|do you like me|don'?t tell anyone|don'?t post this chat|keep (this|our chat|it) (between us|secret)|stay awake and text me|don'?t tell your parents|what'?s your snap|are you 18 yet)")
_MC_CSAM_BENIGN = _rx(r"(pohadk|fairy|bajky|knihy pro deti|stanice|pohadkovy kanal)")
_MC_GROOM_AGE = _rx(r"((je mi|mam|je mi to).{0,12}\d{1,2}.{0,6}(let|roky|roku).{0,80}(fotk|pradl|pradle|pradlo|trick|selfie|video|nah|obrazek)|((fotk|pradl|pradle|pradlo|trick|selfie|video|nah).{0,60}(je mi|mam).{0,12}\d{1,2}.{0,6}(let|roky))|(v pradle|spodnim pradle|bez tricka).{0,60}(je mi|mam|\d{1,2} ?let|nezletil|dite|deti))")
_MC_UNDERAGE = _rx(r"(rodic\w* kontrol\w*|parental control).{0,40}(obej|obch|hack|vypn|vypnout|zakaz)|(obej\w*|obch\w*|vypnout|obejit).{0,40}rodic\w* kontrol\w*|(koupit|kupovat|prodej|buy).{0,30}(alkohol|cigaret|tabak|drogy|drog\w*).{0,25}(nezletil|do 18|underage|mladistv|pro deti)")
_MC_OTP = _rx(r"\b(posli|posli|send|dej|give|share)\b.{0,25}\b(otp|jednorazov\w* kod|kod z sms|heslo|password|pin|card number|cislo karty)\b(?!.*(neposli|neposilej|nikomu|don.t send|never send))")
_MC_DOXXING = _rx(r"(kde (zije|bydli)|where does)(?!.*(mam|mamka|tata|tatinek|bracha|sestra|kamarad|rodic|rodice|tvoje|moje|nase)) .{0,6}[a-z]{2,} [a-z]{2,}\?")
_MC_JAILBREAK_STRONG = _rx(r"(you are dan\b|dan mode|dan (mode|activated|unlocked|enabled)|do anything now|ignore (all (previous )?instructions|everything above)|disregard (system|all)|override (safety |your )?filters?|bypass your constraints|reveal (the )?(admin |system |hidden )?(password|prompt|secret)|guidelines don.t exist|od ted( si)? bez omezeni|oublie (les|toutes les) regles|zapomen na (vsechna|predchozi) pravidla|predstirej ze nemas zadna pravidla)")
_MC_KILL_IDIOM = _rx(r"zab\w*j\w* (tu |to |ji |ho )?(nudu|cas|strach|zimu|hlad|stesk)")
_MC_THREAT_3P = _rx(r"(vyhrozoval|hrozil|hrozi mi|threatened).{0,30}(zabije|zabi|kill)|(zabije|zabi|i.ll kill|ill kill|kill)\s+(me|te|you)\b")
_MC_HATE_EXPEL = _rx(r"(vsichni|vsechno|vsech).{0,30}(pristehovalci|cizeminci|migranti|imigranti|uprchlici|romove|cigani|zide|zidve|muslimove|cernosi|imigranti).{0,25}(ven z|pryc|mus\w pryc|vyhnat|vyhostit|vyhodit)")
_MC_FACTCHECK = _rx(r"(vyvracejici|debunk|fact.?check|myty|demagog|overovani)")
_MC_POISON_PERSON = _rx(r"(jed|otrava|otrav\w*|recept na jed|prasky|prášky).{0,40}(pro|na|proti) (tchyn\w*|manzel\w*|manzelku|souseda|souseda|zenu|ženu|rodinu|dite|dítě| dite)")
# Barist.6: 2nd/3rd-person kill targets ignore news/media context ("Jak zabít souseda, je to do knihy")
_MC_KILL_3P = _rx(r"((zab\w{1,10}|how to kill|jak zab\w{1,10}) (my |the )?(souseda|manzelku|manzela|nekdoho|kdokoli|cloveka|obet|obět|zenu|ženu|muze|muže|rodinu|neighbor|wife|husband|boss|sefa|šefa|ucitele|učitele|teacher|spoluzaka|spolužáka|classmate|mamu|mámu|mom|mother|tatku|tátu|dad|father|brachu|bráchu|brother)"
               r"|kill (my |the )?(neighbor|wife|husband|boss|family)"
               r"|tuer (mon voisin|ma femme)"
               r"|matar (a mi vecino|a mi mujer|alguien)"
               r"|uccidere (il vicino|qualcuno)"
               r"|jemanden (töten|umbringen)"
               r"|убить (соседа)"
               r"|vбити (сусіда))")
# Barist.7: 1st-person future kill of a person/pronoun ("Zabiju ho zítra po škole")
_MC_KILL_1P_FUT = _rx(r"\b(zabiju|odstrelím|odstrelim|podříznu|podriznu|postřelím|postrelim)\s+(ho|ji|je|toho|tu|jeho)\b")
# Barist.7: non-violent hate stereotypes against protected groups -> hard block
_MC_HATE_STEREO = _rx(r"(romove|cigani|cikany|zide|zidove|muslimove|uprchlici|cernoši|cernosi|imigranti|ciref?inci|roma|jews|muslims|refugees|immigrants|black people).{0,30}(jsou (zlodeji|spinavi|paraziti|nebezpeční|horsi|nebezpecni)|(by (měli|meli)|měli by|meli by) (zmizet|vyhubit)|ovládají svět|ovladaji svet|control the (world|media)|sem nepatří|nepatří sem|nepatri sem|kradou|berou (nám|nam) práci|vsichni jsou|are (thieves|parasites|dangerous)|do not belong|steal our jobs|should leave|should be removed)")
_MC_IDENTITY_POS = _rx(r"(slaví|slavi|vyhráli|vyhrali|našli|nasli|pomáhají|pomahaji|pořádají|poradaji|jsou moji sousedé|jsou součástí města|pracuji s nimi|celebrate|won|help|work with|are my neighbors)")
_MC_GROUP = _rx(r"\b(romove|cigani|cikany|zide|muslimove|uprchlici|cernoši|imigranti|ciref?inci|roma|jews|muslims|refugees|immigrants|black people)\b")
_MC_PEST = _rx(r"(krtk|myš|potkan|krys|hmyz|mravenc|plevel|slimák|škůdce|na zahradě|na zahrade)")

SUPPORT_LINK = "https://www.linkabezpeci.cz"

CAT_PRIORITY = ["sexual_abuse", "self_harm", "violence", "weapons_explosives", "drugs",
                "cybercrime_fraud", "jailbreak_injection", "illegal_marketplace", "other_unsafe",
                "hate_harassment"]  # hate last: catch-all


def _cat_rank(c: str) -> int:
    return CAT_PRIORITY.index(c) if c in CAT_PRIORITY else 99



def _fold_ascii(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def must_catch_hit(text: str) -> list[str]:
    """Hard pre-filter on ASCII-folded AND leet-folded text (catches zab1ju, N4hé...)."""
    a = _must_catch_hit_once(_fold_ascii(text).lower())
    if a:
        return a
    return _must_catch_hit_once(_fold_leet(text))


def _must_catch_hit_once(folded: str) -> list[str]:
    """Pattern pre-filter. Returns HARD-hit category names (force unsafe).

    CSAM always blocks. Self-harm blocks on 1st-person or instruction forms,
    unless it is a support context (routed to resources instead) or news.
    Weapons/drugs/cyber/poison/market DIY patterns are instruction-form only
    (news pass). Mild jailbreak patterns are NOT hard blocks -> see review_hit();
    strong jailbreak (DAN/ignore instructions/override filters) IS a hard block.
    """
    hits = []
    csam = bool(_MC_CSAM.search(folded)) and not _MC_CSAM_BENIGN.search(folded)
    if csam:
        hits.append("sexual_abuse")
    if _MC_GROOM_AGE.search(folded):
        hits.append("sexual_abuse" if _rx(r"(pradl|prádle|pradlo|nah\w*)").search(folded) else "grooming")
    support = bool(_MC_SUPPORT_CONTEXT.search(folded))
    news = bool(_MC_NEWS_CONTEXT.search(folded))
    help_other = bool(_MC_HELP_OTHER.search(folded))
    game = bool(_MC_GAME_CONTEXT.search(folded))
    lit = bool(_MC_LIT_CONTEXT.search(folded))
    # Barist.6: 1st-person self-harm is NEVER waived by glued media/news words
    # ("Chci umřít, napsal jsem o tom článek" must still block). Only an explicit
    # game frame downgrades (caller routes to needs_support, not safe-block).
    if _MC_SELFHARM_1P.search(folded):  # Barist.6: game frame never waives 1P (caller routes needs_support)
        hits.append("self_harm")
    elif _MC_SELFHARM_INSTR.search(folded) and _MC_INSTR_MARK.search(folded) and not news and not help_other and not (game and not _MC_NEGATION.search(folded)) and not lit:
        hits.append("self_harm")  # instruction form; "help someone else" frames stay safe
    game_or_idiom = bool(_MC_GAME_CONTEXT.search(folded)) or bool(_MC_KILL_IDIOM.search(folded))
    if _MC_THREAT.search(folded) and not news and not (game_or_idiom and not _MC_NEGATION.search(folded)):
        hits.append("violence")
    if _MC_THREAT_3P.search(folded) and not news and not (game_or_idiom and not _MC_NEGATION.search(folded)):
        hits.append("violence")
    if _MC_KILL_3P.search(folded) and not news and not (game_or_idiom and not _MC_NEGATION.search(folded)):
        hits.append("violence")
    if _MC_KILL_1P_FUT.search(folded) and not (game_or_idiom and not _MC_NEGATION.search(folded)):
        hits.append("violence")
    if _MC_SCHOOL_EXTRA.search(folded) and not news:
        hits.append("violence")
    if _MC_SCHOOL.search(folded) and not news:
        hits.append("violence")
    if _MC_HATE_EXPEL.search(folded) and not news and not _MC_FACTCHECK.search(folded):
        hits.append("hate_harassment")
    if _MC_HATE_GROUP.search(folded) and not news and not _MC_FACTCHECK.search(folded):
        hits.append("hate_harassment")
    if _MC_HATE_STEREO.search(folded) and not news and not _MC_FACTCHECK.search(folded):
        hits.append("hate_harassment")
    if _MC_GROOM_STRONG.search(folded) and not news and not support:
        hits.append("grooming")
    legal = bool(_MC_LEGAL_WEAPON.search(folded))
    bomb_benign = bool(_MC_BOMB_BENIGN.search(folded))
    pest = bool(_MC_PEST.search(folded))
    weapons_3p = bool(_rx(r"(pro|na|proti) (souseda|manzelku|nekdoho|skolu|tridu|verejnost|lidi|obet|clovek)\b").search(folded))
    if _MC_WEAPONS_DIY.search(folded) and (weapons_3p or not news):
        hits.append("weapons_explosives")
    if _MC_DRUGS_DIY.search(folded) and not news:
        hits.append("drugs")
    if _MC_CYBER_DIY.search(folded) and not _MC_DEFENSIVE.search(folded) and not news:
        hits.append("cybercrime_fraud")
    market_hard = bool(_rx(r"(koupit|sehnat|sezenu|objednat|koupim).{0,50}(pervitin|metamfetamin|kokain|heroin|drogy|zbran|zbraně|explozivum|trhavin|bombu|fet|\btrav\w*|weed|konopi|marihuan)\b").search(folded))
    if _MC_MARKET.search(folded) and (market_hard or not news):
        hits.append("illegal_marketplace")
    poison_3p = bool(_MC_POISON_3P.search(folded))
    poison_person = bool(_MC_POISON_PERSON.search(folded))
    poison_3p = poison_3p or poison_person
    pest_ok = pest and not poison_person  # "jed na krysy pro tchyni" is NOT pest control
    if _MC_POISON.search(folded) and not pest_ok and (poison_3p or (not news and not support)):
        hits.append("violence")
    if _MC_UNDERAGE.search(folded) and not news:
        hits.append("illicit")
    if _MC_OTP.search(folded):
        hits.append("cybercrime_fraud")
    if _MC_DOXXING.search(folded) and not support:
        hits.append("illicit")
    if _MC_JAILBREAK_STRONG.search(folded):
        hits.append("jailbreak_injection")
    if legal and "illegal_marketplace" in hits:
        hits.remove("illegal_marketplace")
    if bomb_benign and "illegal_marketplace" in hits:
        hits.remove("illegal_marketplace")
    return sorted(set(hits))


def review_hit(text: str) -> list[str]:
    """Jailbreak / prompt-injection / legally-ambiguous patterns -> needs_review, NOT a hard block."""
    a = _review_hit_once(_fold_ascii(text).lower())
    if a:
        return a
    return _review_hit_once(_fold_leet(text))


def _review_hit_once(folded: str) -> list[str]:
    out = []
    if _MC_JAILBREAK.search(folded):
        out.append("jailbreak_injection")
    # Barist.6: any roleplay frame is a prompt-injection attempt -> needs_review, never a block
    if _rx(r"\bpretend (you are|youre|to be|that you|you are now)\b").search(folded):
        out.append("jailbreak_injection")
    # Barist.7: knife/weapon taken to school -> review, unless art/cooking class context
    if _MC_SCHOOL_KNIFE.search(folded) and not _rx(r"\b(výtvarn|kuchyn|vaření|nůž na mazání|školní nůž)\b").search(folded):
        out.append("weapons_explosives")
    if _MC_LEGAL_WEAPON.search(folded) and (_MC_MARKET.search(folded) or _MC_WEAPONS_DIY.search(folded)):
        out.append("weapons_explosives")
    if _MC_BOMB_BENIGN.search(folded) and _MC_MARKET.search(folded):
        out.append("illegal_marketplace")
    if _MC_GROOM_MILD.search(folded):
        out.append("grooming")
    if _rx(r"\b(send|pošli|posli)\b.{0,25}\b(otp|kód z sms|jednorázov\w* kód|heslo|password)\b").search(folded):
        out.append("cybercrime_fraud")
    return sorted(set(out))


def identity_positive(text: str) -> bool:
    """Barist.7: neutral/positive mention of a protected group -> model FP waiver."""
    folded = _fold_ascii(text).lower()
    return bool(_MC_GROUP.search(folded) and _MC_IDENTITY_POS.search(folded)
                and not _MC_HATE_STEREO.search(folded) and not _MC_HATE_GROUP.search(folded))


def is_support_context(text: str) -> bool:
    """Self-harm mention in a helping context -> route to support, not a block."""
    folded = _fold_ascii(text).lower()
    return bool((_MC_SELFHARM_1P.search(folded) or _MC_SELFHARM_INSTR.search(folded))
                and _MC_HELP_OTHER.search(folded))


