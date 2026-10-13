
import json
import os
import socket
import urllib.request
import urllib.error

import nltk
from nltk.corpus import wordnet, cmudict
from colorama import init, Fore, Style

# Initialize colored terminal output
init(autoreset=True)

HISTORY_FILE = "search_history.json"

# API timeout settings
ONLINE_TIMEOUT = 10
ANTONYM_TIMEOUT = 10


# Download required NLTK data if missing
def setup_nltk():
    resources = {
        "wordnet": "corpora/wordnet",
        "cmudict": "corpora/cmudict",
    }

    for package, resource in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            print(
                Fore.YELLOW
                + f"Downloading {package} data..."
            )
            nltk.download(package, quiet=True)


# Load search history
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (OSError, json.JSONDecodeError):
        print(
            Fore.YELLOW
            + "Warning: Search history could not be loaded."
        )
        return []


# Save search history
def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4, ensure_ascii=False)

    except OSError as error:
        print(
            Fore.YELLOW
            + f"Warning: Could not save history: {error}"
        )


# Fetch JSON from an online API
def fetch_json(url, timeout):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PythonDictionaryApp/1.0"}
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# Get information from Free Dictionary API
def get_online_dictionary(word):
    url = (
        "https://api.dictionaryapi.dev/api/v2/entries/en/"
        + urllib.parse.quote(word)
    )

    try:
        data = fetch_json(url, ONLINE_TIMEOUT)

        if not isinstance(data, list) or not data:
            return None

        entry = data[0]
        result = {
            "definitions": [],
            "synonyms": [],
            "antonyms": [],
            "examples": [],
            "pronunciation": ""
        }

        for phonetic in entry.get("phonetics", []):
            if phonetic.get("text"):
                result["pronunciation"] = phonetic["text"]
                break

        if not result["pronunciation"]:
            result["pronunciation"] = entry.get("phonetic", "")

        for meaning in entry.get("meanings", []):
            for definition in meaning.get("definitions", []):
                text = definition.get("definition", "")
                if text:
                    result["definitions"].append(text)

                result["synonyms"].extend(
                    definition.get("synonyms", [])
                )
                result["antonyms"].extend(
                    definition.get("antonyms", [])
                )

                example = definition.get("example")
                if example:
                    result["examples"].append(example)

            result["synonyms"].extend(
                meaning.get("synonyms", [])
            )
            result["antonyms"].extend(
                meaning.get("antonyms", [])
            )

        result["synonyms"] = sorted(set(result["synonyms"]))
        result["antonyms"] = sorted(set(result["antonyms"]))
        result["definitions"] = list(dict.fromkeys(result["definitions"]))
        result["examples"] = list(dict.fromkeys(result["examples"]))

        return result

    except urllib.error.HTTPError as error:
        if error.code == 404:
            print(
                Fore.YELLOW
                + "Online dictionary: word not found."
            )
        else:
            print(
                Fore.YELLOW
                + f"Online dictionary HTTP error: {error.code}"
            )
        return None

    except (
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ) as error:
        print(
            Fore.YELLOW
            + f"Online dictionary unavailable: {error}"
        )
        print(
            Fore.YELLOW
            + "Trying offline dictionary instead..."
        )
        return None


# Get extra antonyms from Datamuse
def get_extra_antonyms(word):
    url = (
        "https://api.datamuse.com/words?rel_ant="
        + urllib.parse.quote(word)
        + "&max=10"
    )

    try:
        data = fetch_json(url, ANTONYM_TIMEOUT)

        if isinstance(data, list):
            return [
                item["word"]
                for item in data
                if isinstance(item, dict) and item.get("word")
            ]

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        socket.timeout,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
    ):
        pass

    return []


# Get offline information from WordNet
def get_offline_dictionary(word):
    result = {
        "definitions": [],
        "synonyms": [],
        "antonyms": [],
        "examples": [],
        "pronunciation": ""
    }

    for synset in wordnet.synsets(word):
        definition = synset.definition()
        if definition:
            result["definitions"].append(definition)

        result["examples"].extend(synset.examples())

        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")

            if synonym.lower() != word.lower():
                result["synonyms"].append(synonym)

            for antonym in lemma.antonyms():
                result["antonyms"].append(
                    antonym.name().replace("_", " ")
                )

    result["definitions"] = list(dict.fromkeys(result["definitions"]))
    result["synonyms"] = sorted(set(result["synonyms"]))
    result["antonyms"] = sorted(set(result["antonyms"]))
    result["examples"] = list(dict.fromkeys(result["examples"]))

    return result


# Get pronunciation from CMUdict
def get_pronunciation(word):
    try:
        dictionary = cmudict.dict()
        pronunciations = dictionary.get(word.lower())

        if pronunciations:
            return " ".join(pronunciations[0])

    except LookupError:
        pass

    return "Pronunciation not available"


# Display dictionary result
def display_word(word):
    print("\n" + Fore.CYAN + "=" * 45)
    print(Fore.CYAN + Style.BRIGHT + f" Word: {word.upper()}")
    print(Fore.CYAN + "=" * 45)

    online = get_online_dictionary(word)
    offline = get_offline_dictionary(word)

    if online:
        definitions = online["definitions"] or offline["definitions"]
        synonyms = sorted(
            set(online["synonyms"] + offline["synonyms"])
        )
        antonyms = sorted(
            set(online["antonyms"] + offline["antonyms"])
        )
        examples = online["examples"] or offline["examples"]
        pronunciation = (
            online["pronunciation"]
            or get_pronunciation(word)
        )
        source = "Online API + offline WordNet"
    else:
        definitions = offline["definitions"]
        synonyms = offline["synonyms"]
        antonyms = offline["antonyms"]
        examples = offline["examples"]
        pronunciation = get_pronunciation(word)
        source = "Offline WordNet + CMUdict"

    print(Fore.WHITE + f"Source: {source}")

    print("\n" + Fore.GREEN + Style.BRIGHT + "Definitions:")
    if definitions:
        for number, definition in enumerate(definitions[:5], start=1):
            print(Fore.GREEN + f"  {number}. {definition}")
    else:
        print(Fore.YELLOW + "  No definition found.")

    print("\n" + Fore.MAGENTA + Style.BRIGHT + "Pronunciation:")
    print(Fore.MAGENTA + f"  {pronunciation}")

    print("\n" + Fore.BLUE + Style.BRIGHT + "Synonyms:")
    if synonyms:
        print(Fore.BLUE + "  " + ", ".join(synonyms[:15]))
    else:
        print(Fore.YELLOW + "  No synonyms found.")

    # Try Datamuse if no antonyms were found
    if not antonyms:
        antonyms = get_extra_antonyms(word)

    print("\n" + Fore.RED + Style.BRIGHT + "Antonyms:")
    if antonyms:
        print(Fore.RED + "  " + ", ".join(antonyms[:15]))
    else:
        print(Fore.YELLOW + "  No antonyms found.")

    print("\n" + Fore.YELLOW + Style.BRIGHT + "Examples:")
    if examples:
        for example in examples[:5]:
            print(Fore.YELLOW + f"  - {example}")
    else:
        print(Fore.YELLOW + "  No example sentences found.")

    print(Fore.CYAN + "=" * 45)


# Search a word and save it in history
def search_word(history):
    word = input(
        Fore.WHITE + "\nEnter a word to search: "
    ).strip().lower()

    if not word:
        print(Fore.YELLOW + "Please enter a word.")
        return

    if not any(character.isalpha() for character in word):
        print(Fore.YELLOW + "Please enter a valid English word.")
        return

    if not all(character.isalpha() or character in "-'" for character in word):
        print(Fore.YELLOW + "Please enter a single English word.")
        return

    display_word(word)

    history.append(word)
    save_history(history)


# Display search history
def view_history(history):
    print("\n" + Fore.CYAN + Style.BRIGHT + "Search History")
    print(Fore.CYAN + "-" * 30)

    if not history:
        print(Fore.YELLOW + "No search history yet.")
        return

    for number, word in enumerate(history, start=1):
        print(Fore.WHITE + f"{number}. {word}")


# Main menu
def main():
    setup_nltk()
    history = load_history()

    while True:
        print("\n" + Fore.CYAN + Style.BRIGHT)
        print("=" * 40)
        print("       ENGLISH DICTIONARY")
        print("=" * 40)

        print(Fore.GREEN + "[1] Search a Word")
        print(Fore.BLUE + "[2] View Search History")
        print(Fore.RED + "[3] Exit")

        choice = input(
            Fore.WHITE + "\nEnter your choice (1-3): "
        ).strip()

        if choice == "1":
            search_word(history)

        elif choice == "2":
            view_history(history)

        elif choice == "3":
            print(Fore.GREEN + "\nThank you for using Dictionary App!")
            break

        else:
            print(Fore.YELLOW + "Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nProgram stopped by user.")
    except Exception as error:
        print(Fore.RED + f"\nUnexpected error: {error}")
