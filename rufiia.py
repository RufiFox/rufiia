import discord
from discord.ext import commands
import requests
import html2text
import asyncio
import logging
import time
import feedparser
import json
import io
import whisper  # Importation de Whisper pour la reconnaissance vocale
from pydub import AudioSegment
import aiohttp
from TTS.api import TTS  # Importation de Coqui TTS
from river import compose, linear_model, metrics, preprocessing
import nltk
from nltk.stem import SnowballStemmer
import os

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DiscordIA")
logger.setLevel(logging.DEBUG)

# Constantes et configurations
GIPHY_API_KEY = ""
GIPHY_API_URL = "https://api.giphy.com/v1/gifs/search"
NGINX_PROXY_URL = 'https://ia.rufi.fr/api/'
SERPAPI_KEY = ''

SUPPORTED_LANGUAGES = {
    'fr-FR': 'Français (France)',
    'en-US': 'Anglais (États-Unis)',
    'es-ES': 'Espagnol (Espagne)',
    'de-DE': 'Allemand (Allemagne)',
    'it-IT': 'Italien (Italie)',
    'zh-CN': 'Chinois (Mandarin, Chine)',
    'ja-JP': 'Japonais',
    'ru-RU': 'Russe',
}

MODEL_PREFERENCE_FILE = 'model_preferences.json'
GUILD_PREFERENCES_FILE = 'guild_preferences.json'

# Structures de données globales
interactions_data = {}
message_history = {}
command_queue = asyncio.PriorityQueue()
message_counter = 0

# Modèles d'apprentissage
text_model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
text_metric = metrics.MAE()
gif_model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
gif_metric = metrics.MAE()

# --- Fonctions auxiliaires ---
def extract_keywords(text):
    return [word for word in text.lower().split() if len(word) > 3]

def simplify_keywords(text):
    stemmer = SnowballStemmer("french")
    return [stemmer.stem(word) for word in extract_keywords(text)]

def fetch_google_rss_feed(keywords):
    query = '+'.join(keywords)
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:3]]

async def fetch_gif_from_giphy(keywords):
    query = '+'.join(keywords[:3])
    params = {"api_key": GIPHY_API_KEY, "q": query, "limit": 1}
    async with aiohttp.ClientSession() as session:
        async with session.get(GIPHY_API_URL, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data['data']:
                    logger.debug("GIF trouvé pour les mots-clés %s", keywords)
                    return data['data'][0]['images']['original']['url']
    logger.debug("Aucun GIF trouvé pour les mots-clés %s", keywords)
    return None

def html_to_text(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = True
    return h.handle(html_content)

def split_message(text, limit=2000):
    parts = []
    while len(text) > limit:
        split_pos = text[:limit].rfind(' ')
        if split_pos == -1:
            split_pos = limit
        parts.append(text[:split_pos])
        text = text[split_pos:].strip()
    parts.append(text)
    return parts

async def check_ollama_response_stream(response, message, prompt, user_id):
    if response.status == 200:
        waiting_msg = await message.channel.send("Génération en cours...")
        full_content = ""
        async for line in response.content:
            if line.strip():
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_content += chunk['response']
                except Exception as e:
                    logger.error("Erreur lors du décodage JSON: %s", e)
                    continue
        if full_content:
            parts = split_message(full_content)
            for i, part in enumerate(parts):
                if i == 0:
                    await waiting_msg.edit(content=part)
                else:
                    await message.channel.send(part)
            final_message = waiting_msg if len(parts) == 1 else await message.channel.send(parts[-1])
            await add_reactions(final_message)
            message_history[final_message.id] = {'content': full_content, 'prompt': prompt, 'user': user_id}
            global message_counter
            message_counter += 1
            if message_counter >= 5:
                gif_url = await fetch_gif_from_giphy(simplify_keywords(full_content))
                if gif_url:
                    await final_message.channel.send(gif_url)
                    message_counter = 0
            return True, full_content
        else:
            await waiting_msg.edit(content="Aucune réponse générée.")
            return False, "Aucune réponse générée."
    else:
        error_message = f"Erreur de l'API (statut {response.status})"
        logger.error(error_message)
        return False, error_message

async def record_interaction(message_id, user_id, interaction_type, content):
    if message_id not in interactions_data:
        interactions_data[message_id] = {'likes': set(), 'dislikes': set()}
    is_positive = (interaction_type == 'like')
    y = 1 if is_positive else 0
    features = {'length': len(content), 'word_count': len(content.split())}
    prediction = text_model.predict_one(features)
    text_model.learn_one(features, y)
    text_metric.update(y, prediction)
    if is_positive:
        interactions_data[message_id]['likes'].add(user_id)
    else:
        interactions_data[message_id]['dislikes'].add(user_id)
    logger.info("Interaction enregistrée pour le message %s", message_id)

async def add_reactions(message):
    for reaction in ['📋', '🔗', '🔄', '👍', '👎']:
        await message.add_reaction(reaction)

def log_command(command, user_id, prompt):
    with open('ia.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] User {user_id} exécuté la commande '{command}' avec prompt: {prompt}\n")
    logger.info("Commande %s exécutée par l'utilisateur %s", command, user_id)

# --- Gestion des préférences ---
def load_preferences(file_path, default={}):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError:
            logger.error("Fichier %s corrompu ou vide.", file_path)
            return default
    else:
        logger.info("Fichier %s n'existe pas. Création d'un fichier vide.", file_path)
        save_preferences(file_path, default)
        return default

def save_preferences(file_path, preferences):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(preferences, file, ensure_ascii=False, indent=4)

user_model_preferences = load_preferences(MODEL_PREFERENCE_FILE)
guild_preferences = load_preferences(GUILD_PREFERENCES_FILE)

def get_priority(model_name):
    model_sizes = {
        "gemma2:2b": 1, "qwen:0.5b": 2, "qwen:1.8b": 3, "qwen:4b": 4,
        "deepseek-r1:1.5b": 5, "gemma2:9b": 6, "qwen:7b": 7, "deepseek-r1:8b": 8,
        "deepseek-coder:6.7b": 9, "mistral": 10, "TinyLlama:latest": 11,
        "qwen:14b": 12, "deepseek-r1:14b": 13, "phi4:latest": 14,
        "deepseek-r1:latest": 15, "llama3.2:3b": 16
    }
    return model_sizes.get(model_name, 16)

# --- Traitement des commandes ---
class CommandItem:
    def __init__(self, priority, data):
        self.priority = priority
        self.data = data
    def __lt__(self, other):
        return self.priority < other.priority

async def process_commands():
    while True:
        command_item = await command_queue.get()
        user_id, command, message, prompt = command_item.data
        model_name = user_model_preferences.get(user_id, "llama3.2:3b")
        try:
            if command in ['^texte', '^textedepuisinternet']:
                data_to_send = json.dumps({"command": command, "model": model_name, "prompt": prompt, "user": user_id})
                async with aiohttp.ClientSession() as session:
                    async with session.post(NGINX_PROXY_URL + 'generate', data=data_to_send.encode(), headers={'Content-Type': 'application/json'}) as response:
                        is_valid, result = await check_ollama_response_stream(response, message, prompt, user_id)
                        if not is_valid:
                            await message.channel.send(f"Erreur : {result}")
        except Exception as e:
            logger.exception("Erreur lors du traitement de la commande")
            await message.channel.send("Erreur inattendue.")
        finally:
            command_queue.task_done()

# --- Configuration du bot ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="^", intents=intents)

# --- Commandes du bot ---
@bot.command(name='aide')
async def custom_help(ctx):
    help_text = """
    **Commandes disponibles :**
    - `^aide` : Affiche cette aide.
    - `^texte <prompt>` : Génère un texte.
    - `^textedepuisinternet <prompt>` : Génère un texte avec infos internet.
    - `^gif <description>` : Envoie un GIF (GIPHY).
    - `^image <description>` : Envoie une image (Google Images).
    - `^audio-texte [langue]` : Transcrit un fichier WAV avec Whisper.
    - `^texte-audio <texte>` : Convertit texte en audio avec Whisper (via synthèse externe).
    - `^modèlepréférence <modèle>` : Définit le modèle préféré.
    - `^listemodèles` : Liste les modèles disponibles.
    - `^setprefs` : Configure les préférences du serveur (admin).
    """
    await ctx.send(help_text)

@bot.command()
async def texte(ctx, *, prompt="Générer un texte aléatoire"):
    user_id = str(ctx.author.id)
    if user_id not in user_model_preferences:
        await ctx.send("Veuillez définir un modèle avec ^modèlepréférence.")
        return
    log_command('^texte', user_id, prompt)
    await command_queue.put(CommandItem(get_priority(user_model_preferences[user_id]), (user_id, '^texte', ctx.message, prompt)))
    await ctx.send(f"Commande ajoutée pour {ctx.author.mention}.", delete_after=5)

@bot.command()
async def textedepuisinternet(ctx, *, prompt):
    user_id = str(ctx.author.id)
    log_command('^textedepuisinternet', user_id, prompt)
    await command_queue.put(CommandItem(get_priority(user_model_preferences.get(user_id, "llama3.2:3b")), (user_id, '^textedepuisinternet', ctx.message, prompt)))
    await ctx.send(f"Recherche en cours pour {ctx.author.mention}.", delete_after=5)

@bot.command()
async def gif(ctx, *, description):
    log_command('^gif', ctx.author.id, description)
    gif_url = await fetch_gif_from_giphy(extract_keywords(description))
    await ctx.send(gif_url if gif_url else "Aucun GIF trouvé.")

@bot.command()
async def image(ctx, *, query):
    params = {"q": query, "tbm": "isch", "safe": "active", "api_key": SERPAPI_KEY}
    search = GoogleSearch(params)
    results = search.get_dict()
    if "images_results" in results and results["images_results"]:
        image_url = results["images_results"][0]["original"]
        embed = discord.Embed(title=f"Résultat pour '{query}'").set_image(url=image_url)
        await ctx.send(embed=embed)
    else:
        await ctx.send("Aucune image trouvée.")

@bot.command(name='audio-texte')
async def audio_to_text(ctx, language=None):
    if not ctx.message.attachments:
        await ctx.send("Veuillez joindre un fichier audio WAV.")
        return
    attachment = ctx.message.attachments[0]
    if not attachment.filename.endswith('.wav'):
        await ctx.send("Seuls les fichiers WAV sont acceptés.")
        return

    if language not in SUPPORTED_LANGUAGES:
        await ctx.send(f"Langue non spécifiée ou non supportée. Choisissez parmi : {', '.join(SUPPORTED_LANGUAGES.keys())}")
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel
        try:
            response = await bot.wait_for('message', check=check, timeout=30)
            language = response.content.strip()
            if language not in SUPPORTED_LANGUAGES:
                await ctx.send("Langue non supportée.")
                return
        except asyncio.TimeoutError:
            await ctx.send("Temps écoulé. Commande annulée.")
            return

    await ctx.send("Transcription en cours avec Whisper...")
    audio_file = io.BytesIO(await attachment.read())
    audio_file.name = "audio.wav"  # Whisper nécessite un nom de fichier

    # Chargement du modèle Whisper
    model = whisper.load_model("base")  # Options : "tiny", "base", "small", "medium", "large"
    result = model.transcribe(audio_file, language=language[:2])  # Utilise les deux premiers caractères (ex. 'fr')
    full_text = result["text"]

    await ctx.send(f"Transcription ({language}) : {full_text.strip()}")



@bot.command()
async def modèlepréférence(ctx, model_name: str):
    if model_name not in AVAILABLE_MODELS:
        await ctx.send("Modèle non disponible. Voir ^listemodèles.")
        return
    user_id = str(ctx.author.id)
    user_model_preferences[user_id] = model_name
    save_preferences(MODEL_PREFERENCE_FILE, user_model_preferences)
    await ctx.send(f"Modèle défini à {model_name} pour {ctx.author.mention}.")

@bot.command()
async def listemodèles(ctx):
    models_list = "\n".join(f"- **{name}**: {desc}" for name, desc in AVAILABLE_MODELS.items())
    await ctx.send(f"Modèles disponibles :\n{models_list}")

@bot.command()
async def setprefs(ctx):
    if not ctx.author.guild_permissions.administrator and ctx.author != ctx.guild.owner:
        await ctx.send("Seuls les admins peuvent configurer les préférences.")
        return
    await ctx.send("Répondez avec : `models modèle1;modèle2;..., gif oui/non`")
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel
    try:
        msg = await bot.wait_for('message', check=check, timeout=60)
        parts = msg.content.split(',')
        prefs = {}
        for part in parts:
            key_value = part.strip().split()
            if len(key_value) >= 2:
                if key_value[0].lower() == 'models':
                    prefs['text_ia_models'] = [m.strip() for m in ' '.join(key_value[1:]).split(';')]
                elif key_value[0].lower() == 'gif':
                    prefs['gif_command'] = ' '.join(key_value[1:]).lower() == 'oui'
        guild_preferences[str(ctx.guild.id)] = prefs
        save_preferences(GUILD_PREFERENCES_FILE, guild_preferences)
        await ctx.send("Préférences enregistrées.")
    except asyncio.TimeoutError:
        await ctx.send("Temps écoulé.")

# --- Événements du bot ---
@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user or reaction.message.author != bot.user:
        return
    message_id = reaction.message.id
    if message_id in message_history:
        if str(reaction.emoji) == '👍':
            await record_interaction(message_id, user.id, 'like', message_history[message_id]['content'])
        elif str(reaction.emoji) == '👎':
            await record_interaction(message_id, user.id, 'dislike', message_history[message_id]['content'])
        elif str(reaction.emoji) == '🔄' and user.id == message_history[message_id]['user']:
            prompt = message_history[message_id]['prompt']
            await command_queue.put(CommandItem(get_priority(user_model_preferences.get(str(user.id), "llama3.2:3b")), (str(user.id), '^texte', reaction.message, prompt)))
            await reaction.message.channel.send(f"Régénération pour {user.mention}...")

@bot.event
async def on_ready():
    logger.info("%s connecté à Discord!", bot.user)
    bot.loop.create_task(process_commands())
    nltk.download('punkt', quiet=True)

# --- Modèles disponibles ---
AVAILABLE_MODELS = {
    "mistral": "Modèle de langage par Mistral AI.",
    "gemma2:9b": "Pour réponses détaillées.",
    "gemma2:2b": "Pour réponses rapides.",
    "qwen:14b": "Grande capacité.",
    "qwen:7b": "Polyvalent.",
    "qwen:4b": "Réponses succinctes.",
    "qwen:1.8b": "Tâches simples.",
    "qwen:0.5b": "Très léger.",
    "deepseek-r1:8b": "Équilibré.",
    "deepseek-r1:latest": "Dernière version DeepSeek.",
    "phi4:latest": "Performance optimale.",
    "deepseek-r1:1.5b": "Moins de ressources.",
    "deepseek-r1:14b": "Tâches avancées.",
    "deepseek-coder:6.7b": "Génération de code.",
    "TinyLlama:latest": "Ultra-léger.",
    "llama3.2:3b": "Interactions de base."
}

# --- Démarrage du bot ---
bot.run('')
