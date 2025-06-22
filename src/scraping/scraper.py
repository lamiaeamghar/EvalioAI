import requests
from bs4 import BeautifulSoup
import csv
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# List of URLs with their corresponding types and limits
urls_to_scrape = [
    {"url": "https://www.mubawab.ma/fr/sc/appartements-a-vendre", "type": "Appartement", "limit": 15513},
    {"url": "https://www.mubawab.ma/fr/sc/maisons-a-vendre", "type": "Maison", "limit": 2239},
    {"url": "https://www.mubawab.ma/fr/sc/villas-et-maisons-de-luxe-a-vendre", "type": "Villa/Maison de luxe", "limit": 6122},
    {"url": "https://www.mubawab.ma/fr/sc/riads-a-vendre", "type": "Riad", "limit": 904},
    {"url": "https://www.mubawab.ma/fr/sc/locaux-a-vendre", "type": "Local commercial", "limit": 2390},
    {"url": "https://www.mubawab.ma/fr/sc/bureaux-et-commerces-a-vendre", "type": "Bureau/Commerce", "limit": 821},
    {"url": "https://www.mubawab.ma/fr/sc/terrains-a-vendre", "type": "Terrain", "limit": 195},
    {"url": "https://www.mubawab.ma/fr/sc/fermes-a-vendre", "type": "Ferme", "limit": 644},
    {"url": "https://www.mubawab.ma/fr/sc/immobilier-divers-a-vendre", "type": "Autre immobilier", "limit": 39}
]

annonces = []
TOTAL_MAX_ANNONCES = sum(item["limit"] for item in urls_to_scrape)  # Sum of all limits (35054)

def scrape_page(url, property_type, page=1):
    """Scrape a single page of a given URL"""
    formatted_url = f"{url}:p:{page}" if page > 1 else url
    try:
        response = requests.get(formatted_url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Erreur réseau pour {url}: {e}")
        return None

def extract_property_data(card, property_type):
    """Extract property data from a single card"""
    # Titre
    title_tag = card.find("h2", class_="listingTit")
    titre = title_tag.get_text(strip=True) if title_tag and title_tag.find("a") else "N/A"

    # Prix
    price_tag = card.find("span", class_="priceTag")
    prix = price_tag.get_text(strip=True) if price_tag else "N/A"

    # Localisation
    location_tag = card.find("span", class_="listingH3")
    localisation = location_tag.get_text(strip=True).replace(" ", "", 1) if location_tag else "N/A"

    # Détails
    details = {
        "surface": "N/A",
        "pièces": "N/A",
        "chambres": "N/A",
        "salles_de_bains": "N/A"
    }
    for detail in card.find_all("div", class_="adDetailFeature"):
        text = detail.get_text(strip=True)
        if "m²" in text:
            details["surface"] = text
        elif "Pièce" in text:
            details["pièces"] = text
        elif "Chambre" in text:
            details["chambres"] = text
        elif "bain" in text:
            details["salles_de_bains"] = text

    # Caractéristiques
    caracteristiques = []
    for feat in card.find_all("div", class_="adFeature"):
        if feat.find("span"):
            caracteristiques.append(feat.find("span").get_text(strip=True))

    # Description
    desc_tag = card.find("p", class_="listingP")
    description = desc_tag.get_text(strip=True) if desc_tag else "N/A"

    return {
        "type": property_type,
        "titre": titre,
        "prix": prix,
        "localisation": localisation,
        "surface": details["surface"],
        "pièces": details["pièces"],
        "chambres": details["chambres"],
        "salles_de_bains": details["salles_de_bains"],
        "caractéristiques": ", ".join(caracteristiques),
        "description": description
    }

# Main scraping loop
for url_data in urls_to_scrape:
    current_url = url_data["url"]
    property_type = url_data["type"]
    category_limit = url_data["limit"]
    page = 1
    annonces_count_for_type = 0
    
    print(f"\nScraping {property_type} properties (limit: {category_limit}) from {current_url}")
    
    while annonces_count_for_type < category_limit and len(annonces) < TOTAL_MAX_ANNONCES:
        print(f"  Page {page}...")
        soup = scrape_page(current_url, property_type, page)
        if not soup:
            break

        cards = soup.find_all("div", class_=lambda x: x and "listingBox" in x)
        if not cards:
            print("    Plus d'annonces trouvées pour cette catégorie.")
            break

        for card in cards:
            if annonces_count_for_type >= category_limit or len(annonces) >= TOTAL_MAX_ANNONCES:
                break
            
            annonce = extract_property_data(card, property_type)
            annonces.append(annonce)
            annonces_count_for_type += 1

        print(f"    → {len(cards)} annonces trouvées sur cette page ({annonces_count_for_type}/{category_limit} pour cette catégorie)")
        page += 1
        time.sleep(1.5)  # Respectful delay between requests

    if len(annonces) >= TOTAL_MAX_ANNONCES:
        print("\nAtteint le nombre maximum total d'annonces.")
        break

# Export CSV
if annonces:
    fieldnames = ["type"] + [key for key in annonces[0].keys() if key != "type"]  # Ensure 'type' is first column
    with open("../../data/raw_data.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annonces)
    print(f"\n✅ {len(annonces)} annonces sauvegardées dans 'raw_data.csv'.")
else:
    print("\n❌ Aucune donnée à exporter.")