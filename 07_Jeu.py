import streamlit as st
import random
import numpy as np
import pandas as pd
import uuid
from copy import deepcopy
import json
import itertools 
import time
import datetime
# --- NOUVEAUX IMPORTS POUR LE MULTIJOUEUR ---
from supabase import create_client, Client
# ---------------------------------------------

# ---------------- CONFIGURATION & PARAMÈTRES GLOBALES ----------------

# --- Constantes de Jeu ---
BASE_DEMANDS = {"A": 450, "B": 350, "C": 250, "D": 150, "E": 100} 
SALARY_PER_EMP = 2500 
FIXED_COSTS = 5000 
INSURANCE_PER_TRUCK_BASE = 600 
TAX_RATE = 0.15 
BASE_PRICES = {"A": 150, "B": 180, "C": 250, "D": 350, "E": 500} 
MIN_TRUCK_RESALE_RATIO = 0.35
PERF_LOSS_PER_AGE = 0.015 
INDEMNITY_PER_EMP = 1800
TOTAL_CLIENTS = 30
COLIS_TYPES = ["A", "B", "C", "D", "E"]
REFERENCE_SPEED = 80 
MAX_TRIPS = 3 
R_D_COST_BASE = 20000 
FAILLITE_RATIO = 0.8
MAX_LOAN_AGE_BEFORE_SEIZURE = 2
MAX_LOAN_CAPACITY_RATIO = 5 
INTEREST_RATE_PER_TURN = 0.03
MIN_LOAN_PAYMENT_RATIO = 0.15
CAPACITY_UNITS_PER_COLIS = {"A": 1.0, "B": 1.5, "C": 2.0, "D": 3.0, "E": 5.0}

TRUCK_MODELS = [
    {"id": "M1 (Lent)", "speed": 80, "capacity": 25, "price": 25000, "maintenance": 500, "purchase_price": 25000, "age": 0},
    {"id": "M2 (Moyen)", "speed": 100, "capacity": 35, "price": 40000, "maintenance": 900, "purchase_price": 40000, "age": 0},
    {"id": "M3 (Rapide)", "speed": 120, "capacity": 45, "price": 60000, "maintenance": 1300, "purchase_price": 60000, "age": 0}
]

R_D_TYPES = {
    "Logistique": {"cost": R_D_COST_BASE, "effect": "Améliore l'efficacité des camions (Capacité +5%).", "boost_value": 0.05},
    "CyberSécurité": {"cost": R_D_COST_BASE * 1.2, "effect": "Protège des cyberattaques et des pannes IT."},
    "Carburant": {"cost": R_D_COST_BASE * 0.9, "effect": "Protège des pénuries et fluctuations des coûts de carburant."}
}

EVENT_LIST = [
    {"name": "Piratage Bancaire Majeur", "type": "Cyber", "text": "Une faille de sécurité majeure affecte la confiance des clients.", "rep_penalty": 0.8, "rep_penalty_uncovered": 0.65},
    {"name": "Grève des Transporteurs", "type": "Logistique", "text": "La demande est réduite de 25%.", "market_effect": 0.75, "market_bonus_covered": 1.10},
    {"name": "Pénurie de Carburant", "type": "Carburant", "text": "Le prix du carburant double. Les coûts fixes et variables augmentent de 50%.", "cost_increase": 1.5, "cost_protection_covered": 1.1},
    {"name": "Nouvelle Route Commerciale", "type": "Market", "text": "Une nouvelle route s'ouvre. Demande accrue de 25%.", "market_effect": 1.25},
    {"name": "Changement de Réglementation", "type": "Reglementation", "text": "Nouvelles normes de sécurité. Frais imprévus plus élevés.", "rep_penalty": 0.95, "unforeseen_cost_mod": 1.5}
]

# ---------------- CONFIGURATION SUPABASE ----------------

# NOTE: VOUS DEVEZ CRÉER UN FICHIER .streamlit/secrets.toml avec vos clés Supabase:
# [supabase]
# url = "VOTRE_URL_SUPABASE"
# key = "VOTRE_CLE_SERVICE_ANON"

@st.cache_resource
def init_supabase():
    """Initialise le client Supabase."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError:
        st.error("🚨 Erreur de configuration: Les clés Supabase sont manquantes dans `.streamlit/secrets.toml`.")
        st.stop()

supabase: Client = init_supabase()

# ---------------- FONCTIONS DE SYNCHRONISATION MULTIJOUEUR ----------------

def save_game_state_to_db(game_id, game_state):
    """Sauvegarde l'état complet du jeu dans Supabase."""
    data_to_save = {
        "game_id": game_id,
        "state_json": json.dumps(deepcopy(game_state), default=str), 
        "turn": game_state.get('turn', 1),
        "updated_at": datetime.datetime.now().isoformat() # <--- LIGNE CORRIGÉE
    }
    
    response = supabase.table("games").upsert(data_to_save).execute()
    return response

def load_game_state_from_db(game_id):
    """Charge l'état complet du jeu depuis Supabase."""
    try:
        response = supabase.table("games").select("state_json").eq("game_id", game_id).single().execute()
    except Exception:
        return None # Gestion des erreurs si la partie n'existe pas ou erreur de connexion
        
    if response.data:
        loaded_state = json.loads(response.data["state_json"])
        return loaded_state
    return None

def update_game_chat(game_id, player_name, message):
    """Ajoute un message au chat de la partie."""
    new_message = {
        "game_id": game_id,
        "sender": player_name,
        "message": message,
        "timestamp": time.strftime("%H:%M:%S", time.localtime())
    }
    supabase.table("chat_messages").insert(new_message).execute()

def load_game_chat(game_id):
    """Charge les 10 derniers messages du chat."""
    response = supabase.table("chat_messages").select("sender, message, timestamp").eq("game_id", game_id).order("timestamp", desc=True).limit(10).execute()
    return response.data if response.data else []


# ---------------- GESTION DE L'ÉTAT DU JEU (Initialisation/Adaptation) ----------------

def _new_truck(model):
    new_truck = deepcopy(model)
    new_truck["uuid"] = str(uuid.uuid4())
    return new_truck

def generate_player_names(num_ia):
    names = []
    # Noms des joueurs IA
    ia_letters = itertools.cycle('BCDEFGHIJKLMNOPQRSTUVWXYZ')
    for i in range(num_ia):
        names.append(f"Ent. {next(ia_letters)} (IA)")
    return names

def initialize_game_state(host_player_name, num_ia_players):
    """Crée l'état initial du jeu avec l'hôte et les IAs."""
    game_id = f"GAME-{uuid.uuid4().hex[:6].upper()}"
    
    ia_names = generate_player_names(num_ia_players)
    player_names = [host_player_name] + ia_names
    
    game_state = {
        "game_id": game_id,
        "turn": 1,
        "market_trend": 1.0,
        "backlog_packages": {t: 0 for t in COLIS_TYPES},
        "event_history": [],
        "current_event": {"name": "Initialisation", "text": "Le jeu commence.", "type": "None"},
        "players": [],
        "num_ia_players": num_ia_players,
        "host_name": host_player_name,
        "current_human_player_index": 0,
        "actions_this_turn": {},
        "players_ready": {host_player_name: False}, # Nouveau pour suivre les validations
        "game_ready": True
    }
    
    for name in player_names:
        is_human = name == host_player_name
        if not name.startswith("Ent. ") and name != host_player_name:
            is_human = True # Marque tous les joueurs rejoignant la partie comme humains
            
        game_state["players"].append({
            "name": name,
            "is_human": is_human,
            "money": 50000,
            "loan": 0,
            "loan_age": 0,
            "reputation": 1.0,
            "employees": 5,
            "trucks": [_new_truck(TRUCK_MODELS[0]) for _ in range(2)],
            "prices": deepcopy(BASE_PRICES),
            "active": True,
            "can_recover": True,
            "rd_boost_log": 0,
            "rd_investment_type": "Aucun",
            "history": ["Initialisation du jeu."],
            "delivered_packages_total": {t: 0 for t in COLIS_TYPES},
            "income": 0, "expenses": 0, "asset_value": 0, "total_capacity": 0,
        })

    # Sauvegarde initiale dans la BDD
    save_game_state_to_db(game_id, game_state)
    
    return game_state

# ---------------- FONCTIONS DE CALCUL (Inchangées) ----------------

# (Insérez ici les fonctions de calcul d'origine)
def calculate_player_capacity(player_data):
    total_capacity = 0
    log_rd_boost = player_data.get("rd_boost_log", 0)
    
    for truck in player_data["trucks"]:
        if not isinstance(truck, dict) or 'id' not in truck: continue 
        
        perf_factor = 1.0 - (truck["age"] * PERF_LOSS_PER_AGE)
        perf_factor = max(0.6, perf_factor) 
        effective_speed = truck["speed"] * perf_factor
        effective_capacity = truck["capacity"] * perf_factor * (1 + log_rd_boost)
        trip_multiplier = min(MAX_TRIPS, effective_speed / REFERENCE_SPEED) 
        total_capacity += int(effective_capacity * trip_multiplier)
        
    return total_capacity

def calculate_asset_value(player_trucks):
    total_value = 0
    for truck in player_trucks:
        if not isinstance(truck, dict) or 'id' not in truck: continue
        
        current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
        resale = max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value)
        total_value += resale
    return int(total_value)

def poisson_market(base, trend=1.0):
    return int(np.random.poisson(max(0, base * trend)))

def generate_client_orders(game_state):
    package_orders = {t: 0 for t in COLIS_TYPES}
    
    for t in COLIS_TYPES:
        package_orders[t] += game_state["backlog_packages"].get(t, 0)
    
    for _ in range(TOTAL_CLIENTS * 2): 
        types_chosen = random.choices(COLIS_TYPES, k=random.randint(1, 3))
        for t in types_chosen:
            package_orders[t] += random.randint(1, 5)

    for t in COLIS_TYPES:
        package_orders[t] += poisson_market(BASE_DEMANDS.get(t, 0), game_state["market_trend"]) 
        
    if "market_effect" in game_state["current_event"]:
        for t in package_orders:
            package_orders[t] = int(package_orders[t] * game_state["current_event"]["market_effect"])

    capacity_required = {
        t: int(package_orders[t] * CAPACITY_UNITS_PER_COLIS.get(t, 1.0)) 
        for t in COLIS_TYPES
    }
    
    return capacity_required

def calculate_competition_score(p, t):
    player_exec_capacity = p["total_capacity"]
    price_score = p["prices"].get(t, BASE_PRICES.get(t, 500)) * 0.4
    rep_score = 800 / max(1, p["reputation"])
    cap_factor = 1000 / (player_exec_capacity + 1)
    
    total_score = price_score + rep_score + cap_factor
    attractiveness_weight = 1.0 / (total_score * total_score)
    
    return attractiveness_weight, player_exec_capacity

def distribute_clients(market_capacity_demand, players, game_state):
    allocation_capacity = {p["name"]: {t: 0 for t in COLIS_TYPES} for p in players}
    current_package_backlog = {t: 0 for t in COLIS_TYPES}
    active_players = [p for p in players if p["active"]]
    
    player_data = {}
    for p in active_players:
        p_cap = calculate_player_capacity(p)
        p["total_capacity"] = p_cap
        player_data[p["name"]] = {
            "player": p,
            "max_capacity": p_cap,
            "current_allocation_total": 0,
            "scores": {t: calculate_competition_score(p, t)[0] for t in COLIS_TYPES}
        }
    
    for t in COLIS_TYPES:
        qty_capacity_remaining = market_capacity_demand.get(t, 0)
        colis_size = CAPACITY_UNITS_PER_COLIS.get(t, 1.0)
        
        if qty_capacity_remaining == 0: continue
        unit_size = max(1, qty_capacity_remaining // 4) 
        
        while qty_capacity_remaining > 0:
            scores_and_weights = []
            for p_name, data in player_data.items():
                p = data["player"]
                cap_remaining_global = data["max_capacity"] - data["current_allocation_total"]
                
                if cap_remaining_global > 0:
                    scores_and_weights.append({
                        "player": p, 
                        "weight": data["scores"].get(t, 0),
                        "cap_remaining_global": cap_remaining_global
                    })

            total_market_weight = sum(item["weight"] for item in scores_and_weights)
            
            if not scores_and_weights or total_market_weight == 0:
                break
            
            weights = [item["weight"] for item in scores_and_weights]
            chosen_items = random.choices(scores_and_weights, weights=weights, k=1)
            
            if chosen_items:
                chosen_item = chosen_items[0]
                p = chosen_item["player"]
                cap_remaining = chosen_item["cap_remaining_global"]
                p_name = p["name"]
                
                capacity_to_distribute = min(unit_size, qty_capacity_remaining) 
                deliverable_capacity = min(capacity_to_distribute, cap_remaining)
                
                if deliverable_capacity > 0:
                    allocation_capacity[p_name][t] += deliverable_capacity
                    qty_capacity_remaining -= deliverable_capacity
                    player_data[p_name]["current_allocation_total"] += deliverable_capacity
                else:
                    pass
            else:
                break
                
        capacity_unallocated = max(0, qty_capacity_remaining)
        packages_unallocated = int(capacity_unallocated / colis_size)
        current_package_backlog[t] += packages_unallocated
        
    for t in current_package_backlog:
        game_state["backlog_packages"][t] = min(20, current_package_backlog[t])
            
    return allocation_capacity

def trigger_random_event(game_state):
    if random.random() < 0.4: 
        event = random.choice(EVENT_LIST)
        game_state["current_event"] = event
        game_state["event_history"].append(f"Tour {game_state['turn']}: {event['name']} - {event['text']}")
        
        if event.get("rep_penalty") and event["type"] not in ["Cyber", "Logistique", "Carburant"]:
            for p in game_state["players"]:
                if p["active"]:
                    p["reputation"] = max(0.5, p["reputation"] * event["rep_penalty"])
    else:
        game_state["current_event"] = {"name": "Aucun", "text": "Un tour normal.", "type": "None"}


# ---------------- LOGIQUE DE JEU INTÉGRÉE (SIMULATE_TURN) ----------------

def simulate_turn_streamlit(game_state, actions_dict):
    """
    Exécute un tour de simulation en utilisant l'état stocké dans game_state
    et les actions des joueurs passées dans actions_dict.
    """
    
    # --- PHASE PRÉ-TOUR ---
    trigger_random_event(game_state)
    current_event = game_state["current_event"]
    event_info = f"🌪️ Événement du Tour: {current_event['name']} - {current_event['text']}"
    
    # 1. Actions IA
    for i, p in enumerate(game_state["players"]):
        if not p["is_human"]:
            ia_action = get_ia_actions(p)
            actions_dict[p["name"]] = ia_action
    
    market_capacity_demand = generate_client_orders(game_state) 

    # --- PHASE D'APPLICATION DES ACTIONS DÉCIDÉES PAR LES JOUEURS ---
    for i, p in enumerate(game_state["players"]):
        
        p["history"] = [event_info]
        action = actions_dict.get(p["name"], {"prices": p["prices"]}).copy()
        
        p["prices"] = action.get("prices", p["prices"])
        p["rd_boost_log"] = p.get("rd_boost_log", 0) 
        p["rd_investment_type"] = action.get("rd_type", "Aucun")
        p["asset_value"] = calculate_asset_value(p["trucks"])

        # 0. Gestion des faillites (Vente d'actifs pour récupérer)
        if not p["active"] and not p.get("can_recover", True):
            p["history"].append("🚨 Entreprise liquidée. Aucune action possible.")
            continue
            
        if not p["active"] and p.get("can_recover", True):
            if "sell_trucks" in action:
                for model_id, qty in action["sell_trucks"].items():
                    if qty > 0:
                        trucks_to_sell = [t for t in p["trucks"] if t.get("id") == model_id][:qty]
                        for truck in trucks_to_sell:
                            if truck in p["trucks"]:
                                p["trucks"].remove(truck)
                                current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10) 
                                resale = int(max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                                p["money"] += resale
                                p["history"].append(f"Vente (Faillite): {truck['id']} (+{resale:,} €)".replace(",", " "))
            
            if p["money"] >= 0:
                asset_val = calculate_asset_value(p["trucks"])
                if p["loan"] / max(1, asset_val) < FAILLITE_RATIO:
                    p["active"] = True
                    p["history"].append("Sortie de Faillite! Solde positif et dette sous contrôle.")
            
            if not p["active"]: 
                p["history"].append("Faillite temporaire: doit vendre plus ou rembourser dette.")
                game_state["players"][i] = p
                continue 
        
        # A. Prêts bancaires
        loan_amount = action.get("loan_request", 0)
        loan_payment = action.get("loan_payment", 0)
        
        if loan_amount > 0:
            asset_value = calculate_asset_value(p["trucks"])
            max_loan = asset_value * MAX_LOAN_CAPACITY_RATIO
            if p["loan"] + loan_amount <= max_loan:
                p["money"] += loan_amount
                p["loan"] += loan_amount
                p["loan_age"] = 0 
                p["history"].append(f"Prêt accordé : +{loan_amount:,} €".replace(",", " "))
            else:
                p["history"].append(f"Prêt refusé : Capacité max ({max_loan:,} €) dépassée.".replace(",", " "))
        
        if loan_payment > 0:
            payable = min(loan_payment, p["loan"])
            if p["money"] >= payable:
                p["money"] -= payable
                p["loan"] -= payable
                p["history"].append(f"Remboursement de prêt : -{payable:,} €".replace(",", " "))
            else:
                p["history"].append(f"Remboursement refusé : Fonds insuffisants.".replace(",", " "))


        # B. Recherche & Développement
        rd_type_chosen = action.get("rd_type", "Aucun")
        if rd_type_chosen != "Aucun":
            rd_config = R_D_TYPES.get(rd_type_chosen, {})
            rd_cost = rd_config.get("cost", 0)
            
            if p["money"] >= rd_cost:
                p["money"] -= rd_cost
                p["rd_investment_type"] = rd_type_chosen 
                
                if rd_type_chosen == "Logistique":
                    p["rd_boost_log"] += rd_config.get("boost_value", 0)
                    p["history"].append(f"R&D Logistique : Capacité effective +{rd_config.get('boost_value', 0)*100:.0f}% !".replace(",", " "))
                else:
                    p["history"].append(f"R&D Risque ({rd_type_chosen}) : Couverture activée.".replace(",", " "))
            else:
                p["rd_investment_type"] = "Aucun"
                p["history"].append(f"R&D ({rd_type_chosen}) refusée: fonds insuffisants.")
        
        # C. Achat/Vente de Camions
        if "buy_trucks" in action:
            for model_id, qty in action["buy_trucks"].items():
                if qty > 0:
                    model = next(m for m in TRUCK_MODELS if m["id"] == model_id)
                    cost = model["price"] * qty
                    if p["money"] >= cost:
                        p["money"] -= cost
                        for _ in range(qty):
                            p["trucks"].append(_new_truck(model)) 
                        p["history"].append(f"Achat: {qty}x {model['id']} (-{cost:,} €)".replace(",", " "))
                    else:
                        p["history"].append(f"Achat {qty}x {model['id']} refusé: fonds insuffisants.")
        
        if "sell_trucks" in action and p["active"]:
            for model_id, qty in action["sell_trucks"].items():
                if qty > 0:
                    trucks_to_sell = [t for t in p["trucks"] if t.get("id") == model_id][:qty]
                    for truck in trucks_to_sell:
                        if truck in p["trucks"]:
                            p["trucks"].remove(truck)
                            current_value = truck["purchase_price"] * (1 - truck["age"] * 0.10)
                            resale = int(max(truck["purchase_price"] * MIN_TRUCK_RESALE_RATIO, current_value))
                            p["money"] += resale
                            p["history"].append(f"Vente: {truck['id']} (+{resale:,} €)".replace(",", " "))
        
        # D. Publicité
        pub_type = action.get("pub_type", "Aucun")
        if pub_type != "Aucun":
            if pub_type == "Locale": cost, rep_inc = (5000, 0.06)
            elif pub_type == "Nationale": cost, rep_inc = (12000, 0.12)
            elif pub_type == "Globale": cost, rep_inc = (25000, 0.25)
            else: cost, rep_inc = (0, 0)
            
            if cost > 0 and p["money"] >= cost:
                p["money"] -= cost
                p["reputation"] = min(5.0, p["reputation"] * (1 + rep_inc))
                p["history"].append(f"Publicité {pub_type}: Réputation +{rep_inc*100:.0f}% (-{cost:,} €)".replace(",", " "))
            elif cost > 0:
                p["history"].append(f"Publicité {pub_type} refusée: fonds insuffisants.".replace(",", " "))

        # E. Employés
        if "emp_delta" in action and action["emp_delta"] != 0:
            delta = action["emp_delta"]
            if delta > 0:
                p["employees"] += delta
                p["history"].append(f"Embauche: +{delta} employés.")
            elif delta < 0:
                nb_lic = min(-delta, p["employees"])
                if nb_lic > 0:
                    indemnity = nb_lic * INDEMNITY_PER_EMP
                    if p["money"] >= indemnity:
                        p["money"] -= indemnity
                        p["employees"] -= nb_lic
                        p["history"].append(f"Licenciement: -{nb_lic} employés (-{indemnity:,} € d'indemnités).".replace(",", " "))
                    else:
                        p["history"].append(f"Licenciement annulé: fonds insuffisants pour les indemnités.".replace(",", " "))
                        
        game_state["players"][i] = p 

    # 2. Distribution clients et mise à jour de l'état
    active_players_for_distribution = [p for p in game_state["players"] if p["active"]]
    
    allocations_capacity = distribute_clients(market_capacity_demand, active_players_for_distribution, game_state)

    # --- PHASE DE CALCUL DES RÉSULTATS ET VÉRIFICATION DE LA FAILLITE ---
    
    for i, p in enumerate(game_state["players"]):
        
        if "delivered_packages_total" not in p:
             p["delivered_packages_total"] = {t: 0 for t in COLIS_TYPES}

        if not p["active"]: continue
        
        allocated_capacity = allocations_capacity.get(p["name"], {t: 0 for t in COLIS_TYPES})
        delivered_packages = {}
        revenue = 0
        
        for t in COLIS_TYPES:
            colis_size = CAPACITY_UNITS_PER_COLIS.get(t, 1.0)
            packages = int(allocated_capacity.get(t, 0) / colis_size)
            delivered_packages[t] = packages
            revenue += packages * p["prices"].get(t, 0)
            p["delivered_packages_total"][t] = p["delivered_packages_total"].get(t, 0) + packages
        
        # --- APPLICATION DES EFFETS R&D ET ÉVÉNEMENTS ---
        cost_mod_event = current_event.get("cost_increase", 1.0)
        market_mod_event = 1.0
        event_type = current_event["type"]
        rd_type_covered = p.get("rd_investment_type", "Aucun")

        if event_type in ["Cyber", "Logistique", "Carburant"] and rd_type_covered != event_type:
            if event_type == "Carburant": cost_mod_event = current_event.get("cost_increase", 1.0)
            elif event_type == "Logistique": market_mod_event = current_event.get("market_effect", 1.0)
            elif event_type == "Cyber": p["reputation"] = max(0.5, p["reputation"] * current_event.get("rep_penalty_uncovered", 1.0))
        elif event_type in ["Logistique", "Carburant"] and rd_type_covered == event_type:
             if event_type == "Carburant": cost_mod_event = current_event.get("cost_protection_covered", 1.0) 
             elif event_type == "Logistique": market_mod_event = current_event.get("market_bonus_covered", 1.0)
        
        # 1. Gestion de la dette et des intérêts
        interest_paid = 0
        loan_repayment_made_this_turn = action.get("loan_payment", 0)
        min_payment_due = p["loan"] * MIN_LOAN_PAYMENT_RATIO
        
        if p["loan"] > 0:
            interest_paid = p["loan"] * INTEREST_RATE_PER_TURN
            p["loan"] += interest_paid
            
            if loan_repayment_made_this_turn >= min_payment_due: 
                p["loan_age"] = 0
            else:
                p["loan_age"] = p.get("loan_age", 0) + 1

        # 2. Vérification de Saisie Bancaire DÉFINITIVE
        if p.get("loan_age", 0) >= MAX_LOAN_AGE_BEFORE_SEIZURE:
            p["active"] = False
            p["can_recover"] = False
            p["money"] = -100000 
            p["history"].append(f"🔥🔥🔥 **SAISIE BANCAIRE** : L'entreprise est **LIQUIDÉE** (Âge du prêt: {p['loan_age']}).")
            game_state["players"][i] = p
            continue 
            
        # 3. Entretien/Usure
        total_maintenance = 0
        for truck in p["trucks"]:
            if not isinstance(truck, dict) or 'id' not in truck: continue
            
            truck["age"] += 1 
            
            if 'maintenance' in truck:
                truck["maintenance"] = int(truck["maintenance"] * (1 + 0.05)) 
                total_maintenance += truck["maintenance"]

        # 4. Coûts
        unforeseen_mod = current_event.get("unforeseen_cost_mod", 1.0)
        
        salaries = p["employees"] * SALARY_PER_EMP
        insurance = len(p["trucks"]) * INSURANCE_PER_TRUCK_BASE
        taxes = revenue * TAX_RATE
        imprevus = random.randint(0, len(p["trucks"]) * 1000 + salaries // 20) * unforeseen_mod
        
        base_fixed_costs = FIXED_COSTS + total_maintenance
        variable_costs_modified = (base_fixed_costs * cost_mod_event) + (imprevus * cost_mod_event)
        expenses_total = variable_costs_modified + salaries + insurance + taxes + interest_paid
        
        # Mise à jour de l'état financier
        p["income"] = revenue * market_mod_event 
        p["expenses"] = expenses_total
        p["money"] += p["income"] - p["expenses"]
        
        p["delivered_packages"] = delivered_packages
        
        # 5. Faillite après opérations (Faillite TEMPORAIRE)
        asset_val = calculate_asset_value(p["trucks"])
        
        if p["money"] < 0 or (p["loan"] > 0 and p["loan"] / max(1, asset_val) > FAILLITE_RATIO):
            if p["active"]:
                p["active"] = False
                p["can_recover"] = True
                p["history"].append(f"🚨 FAILLITE TEMPORAIRE! Solde négatif ({int(p['money']):,} €) ou dette/actif ({p['loan'] / max(1, asset_val):.2f}) > {FAILLITE_RATIO}. Vendez pour survivre.".replace(",", " "))

        game_state["players"][i] = p

    # 3. Finalisation du tour
    game_state["market_trend"] *= random.uniform(0.85, 1.15)
    
    # NE PAS INCREMENTER LE TOUR ICI. L'INCREMENTATION SE FAIT DANS run_next_turn.
    # game_state["turn"] += 1 
    
    return game_state

# ---------------- LOGIQUE DU JOUEUR IA (AMÉLIORÉE) ----------------

def get_ia_actions(p):
    """Logique d'action plus intelligente pour le joueur IA."""
    current_money = p["money"]
    truck_count = len(p["trucks"])
    capacity = p.get("total_capacity", 0)
    action = {"prices": deepcopy(p["prices"]), "buy_trucks": {}, "sell_trucks": {}}

    monthly_costs_est = (p["employees"] * SALARY_PER_EMP) + (truck_count * INSURANCE_PER_TRUCK_BASE) + FIXED_COSTS

    # --- 1. Gestion de la Faillite/Crise (Priorité Absolue) ---
    if not p["active"] and p.get("can_recover"):
        if current_money < -monthly_costs_est * 2 and truck_count > 1:
            m1_trucks = [t for t in p["trucks"] if t.get("id") == "M1 (Lent)"]
            if m1_trucks:
                action["sell_trucks"] = {"M1 (Lent)": 1}
                return action
        elif current_money < -10000 and p["loan"] == 0:
             action["loan_request"] = 20000

    if not p["active"]: 
        return action

    # --- 2. Stratégie de Croissance (Si Solde Sain) ---
    if current_money > 120000 and truck_count < 15 and capacity < 600:
        if random.random() < 0.3:
            action["buy_trucks"] = {"M3 (Rapide)": 1}
        else:
            action["buy_trucks"] = {"M2 (Moyen)": 1}
        
    # --- 3. Stratégie de R&D et Publicité ---
    if current_money > 80000 and p["rd_boost_log"] < 0.15 and random.random() < 0.2:
        action["rd_type"] = "Logistique"
        
    if p["reputation"] < 0.8 and current_money > 15000:
        action["pub_type"] = "Nationale"
        
    # --- 4. Gestion des Prêts ---
    min_payment_due = p["loan"] * MIN_LOAN_PAYMENT_RATIO
    if p["loan"] > 0 and current_money > 50000 + monthly_costs_est:
        action["loan_payment"] = int(p["loan"] * 0.3)
    elif p["loan"] > 0 and current_money > min_payment_due + monthly_costs_est * 1.5:
        action["loan_payment"] = int(min_payment_due * 1.5)

    # --- 5. Ajustement des Prix ---
    if p["reputation"] > 1.2 and random.random() < 0.1:
          for t in action["prices"]:
              action["prices"][t] = int(action["prices"][t] * 1.05)
              
    elif p["reputation"] < 0.9 and random.random() < 0.1:
          for t in action["prices"]:
              action["prices"][t] = int(action["prices"][t] * 0.95)

    return action

# ---------------- LOGIQUE DE FLUX MULTIJOUEUR ----------------

def run_next_turn(actions_dict):
    """ Lance la simulation du tour, met à jour l'état et le synchronise. """
    
    # 1. Copie de l'état actuel
    current_state = deepcopy(st.session_state)
    
    # 2. Exécution du tour (Met à jour current_state)
    new_state_data = simulate_turn_streamlit(current_state, actions_dict)
    
    # 3. Finalisation du tour
    new_state_data["turn"] += 1
    
    # 4. Réinitialisation des actions et du statut de prêt
    new_state_data["actions_this_turn"] = {}
    new_state_data["players_ready"] = {p["name"]: False for p in new_state_data["players"] if p["is_human"]} 
    
    # 5. Mise à jour de la session Streamlit et sauvegarde dans la BDD
    st.session_state.update(new_state_data)
    save_game_state_to_db(st.session_state.game_id, st.session_state)


# ---------------- INTERFACE UTILISATEUR ET FORMULAIRES ----------------

def show_delivery_summary(players_list):
    """Affiche un tableau récapitulatif des colis livrés par type."""
    delivery_data = []
    
    for p in players_list:
        row = {"Entreprise": p["name"], "Statut": "Actif" if p["active"] else "Faillite"}
        if "delivered_packages_total" not in p:
             p["delivered_packages_total"] = {t: 0 for t in COLIS_TYPES}
             
        row.update(p["delivered_packages_total"])
        row["Total Livré"] = sum(p["delivered_packages_total"].values())
        delivery_data.append(row)
        
    df = pd.DataFrame(delivery_data).set_index("Entreprise")
    st.dataframe(df.sort_values(by="Total Livré", ascending=False))
    
def show_final_results():
    """Affiche le classement final."""
    st.markdown("## 🏆 Classement Final")
    
    # Calcul du score final (Argent + Valeur Actifs - Dette)
    final_data = []
    for p in st.session_state.players:
        score = p['money'] + calculate_asset_value(p['trucks']) - p['loan']
        final_data.append({
            "Entreprise": p['name'],
            "Statut": "Liquidée" if not p.get('can_recover', True) and not p['active'] else ("Actif" if p['active'] else "Faillite"),
            "Trésorerie": int(p['money']),
            "Dette": int(p['loan']),
            "Actifs (Camions)": int(calculate_asset_value(p['trucks'])),
            "Score Final (€)": int(score)
        })

    df = pd.DataFrame(final_data).set_index("Entreprise")
    st.dataframe(df.sort_values(by="Score Final (€)", ascending=False), use_container_width=True)


def get_human_actions_form(player_data, disabled=False):
    """Formulaire d'actions pour le joueur humain."""
    
    actions = {}
    
    current_prices = player_data["prices"]
    current_money = player_data["money"]
    current_loan = player_data["loan"]
    current_emp = player_data["employees"]
    current_rd_type = player_data.get("rd_investment_type", "Aucun")
    
    if not player_data["active"]:
        st.warning(f"Votre statut est 'Faillite'. Vous ne pouvez que vendre des actifs.")
        disabled = False # On autorise la vente pour récupérer

    with st.form(key=f"form_{player_data['name']}", clear_on_submit=False):
        
        # --- Colonnes de Statut ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Trésorerie Actuelle", f"{current_money:,.0f} €")
        col2.metric("Dette Bancaire", f"{current_loan:,.0f} €")
        col3.metric("Réputation", f"{player_data['reputation']:.2f}")

        st.markdown("### 1. Fixer les Prix des Colis")
        
        new_prices = {}
        cols = st.columns(len(COLIS_TYPES))
        for i, t in enumerate(COLIS_TYPES):
            new_prices[t] = cols[i].number_input(f"Prix Colis {t}", min_value=1, value=current_prices.get(t, BASE_PRICES[t]), disabled=disabled)
        actions["prices"] = new_prices

        st.markdown("### 2. Gestion de la Flotte et des Finances")
        
        # Achat/Vente de Camions
        st.subheader("Achats / Ventes de Camions")
        
        buy_actions = {}
        sell_actions = {}
        
        for model in TRUCK_MODELS:
            col_buy, col_sell = st.columns(2)
            
            # Achat
            buy_qty = col_buy.number_input(f"Acheter {model['id']} ({model['price']:,.0f} €)", min_value=0, value=0, key=f"buy_{model['id']}_{player_data['name']}", disabled=disabled)
            buy_actions[model['id']] = buy_qty
            
            # Vente
            trucks_owned = len([t for t in player_data['trucks'] if t['id'] == model['id']])
            sell_qty = col_sell.number_input(f"Vendre {model['id']} (Possédés: {trucks_owned})", min_value=0, max_value=trucks_owned, value=0, key=f"sell_{model['id']}_{player_data['name']}", disabled=False if not player_data["active"] else disabled)
            sell_actions[model['id']] = sell_qty
            
        actions["buy_trucks"] = buy_actions
        actions["sell_trucks"] = sell_actions
            
        # Prêts et Remboursements
        st.subheader("Prêts Bancaires")
        col_loan_req, col_loan_pay = st.columns(2)
        loan_request = col_loan_req.number_input("Demander un Prêt (€)", min_value=0, step=1000, value=0, disabled=disabled)
        loan_payment = col_loan_pay.number_input("Rembourser un Prêt (€)", min_value=0, step=1000, max_value=int(current_loan) if current_loan > 0 else 0, value=0, disabled=disabled)
        actions["loan_request"] = loan_request
        actions["loan_payment"] = loan_payment
        
        # R&D
        st.subheader("Recherche & Développement")
        rd_options = ["Aucun"] + list(R_D_TYPES.keys())
        rd_type = st.selectbox("Investissement R&D (Annuel)", rd_options, index=rd_options.index(current_rd_type) if current_rd_type in rd_options else 0, disabled=disabled)
        actions["rd_type"] = rd_type
        
        # Publicité
        st.subheader("Campagne de Publicité (Réputation)")
        pub_options = ["Aucun", "Locale", "Nationale", "Globale"]
        pub_type = st.selectbox("Type de Publicité", pub_options, disabled=disabled)
        actions["pub_type"] = pub_type
        
        # Employés
        st.subheader("Gestion des Employés (Actuel: " + str(current_emp) + ")")
        emp_delta = st.number_input("Embaucher (+) / Licencier (-)", min_value=-current_emp, value=0, step=1, disabled=disabled)
        actions["emp_delta"] = emp_delta
        
        st.form_submit_button("Pré-Validation (Ne valide pas le tour!)", disabled=True, help="Cliquez sur le bouton bleu principal pour valider le tour")
        
        return actions


def show_chat_sidebar(game_id, player_name):
    """Affiche la section de chat et la synchronisation."""
    st.sidebar.subheader("💬 Chat de la Partie")

    # Champ d'entrée du chat
    with st.sidebar.form("chat_form", clear_on_submit=True):
        message = st.text_input("Message")
        submitted = st.form_submit_button("Envoyer")
        
        if submitted and message:
            update_game_chat(game_id, player_name, message)

    # Affichage des messages
    messages = load_game_chat(game_id)
    chat_box = st.sidebar.container(height=300)
    for msg in reversed(messages):
        chat_box.write(f"**{msg['timestamp']} {msg['sender']}**: {msg['message']}")
    
    # Bouton de synchronisation manuelle
    if st.sidebar.button("🔄 Actualiser le Statut du Jeu", help="Cliquez pour forcer la synchronisation avec la partie en cours."):
        st.info("Synchronisation...")
        loaded_state = load_game_state_from_db(game_id)
        if loaded_state:
             st.session_state.update(loaded_state)
             st.rerun()
        else:
             st.error("Impossible de se synchroniser. Vérifiez la connexion.")

# ---------------- BOUCLE PRINCIPALE ----------------

def main():
    st.set_page_config(layout="wide", page_title="Simulateur de Transport Multijoueur")
    
    # --- 0. Écran de Connexion / Création ---
    if 'game_id' not in st.session_state:
        st.title("🤝 Rejoindre ou Créer une Partie")
        
        # Utiliser un identifiant unique par défaut pour le nom
        if 'user_name' not in st.session_state:
            st.session_state.user_name = f"Joueur-{uuid.uuid4().hex[:4].upper()}"
            
        player_name = st.text_input("Votre Nom/Entreprise (Ex: Joueur H1)", key="current_user_name", value=st.session_state.user_name)
        st.session_state.user_name = player_name
        
        st.subheader("Créer une nouvelle partie")
        num_ia = st.number_input("Nombre d'entreprises IA (concurrents)", min_value=1, max_value=9, value=3)
        
        if st.button("🚀 Créer et Héberger la Partie", type="primary"):
            st.session_state.update(initialize_game_state(player_name, num_ia))
            st.session_state.my_name = player_name
            st.success(f"Partie créée! ID: {st.session_state.game_id}")
            st.rerun()
            
        st.subheader("Rejoindre une partie existante")
        join_id = st.text_input("Entrer l'ID de la partie")
        
        if st.button("🔗 Rejoindre la Partie"):
            loaded_state = load_game_state_from_db(join_id)
            if loaded_state:
                st.session_state.my_name = player_name
                
                # Ajouter le joueur s'il n'est pas déjà dans la partie
                if player_name not in [p['name'] for p in loaded_state['players']]:
                    new_player = {
                        "name": player_name,
                        "is_human": True, "money": 50000, "loan": 0, "loan_age": 0, "reputation": 1.0, 
                        "employees": 5, "trucks": [_new_truck(TRUCK_MODELS[0]) for _ in range(2)],
                        "prices": deepcopy(BASE_PRICES), "active": True, "can_recover": True, "rd_boost_log": 0,
                        "rd_investment_type": "Aucun", "history": ["Rejoint la partie."],
                        "delivered_packages_total": {t: 0 for t in COLIS_TYPES}, "income": 0, 
                        "expenses": 0, "asset_value": 0, "total_capacity": 0,
                    }
                    loaded_state['players'].append(new_player)
                    loaded_state['players_ready'][player_name] = False
                    
                    loaded_state['num_human_players'] = len([p for p in loaded_state['players'] if p['is_human']])
                    save_game_state_to_db(join_id, loaded_state)
                
                st.session_state.update(loaded_state)
                st.session_state.game_id = join_id
                st.rerun()
            else:
                st.error("Partie non trouvée ou ID invalide.")
        return
        
    # ---------------- DÉBUT DE L'INTERFACE DE JEU ----------------
    
    game_id = st.session_state.game_id
    my_name = st.session_state.my_name
    
    # Affichage du chat et du bouton de synchro
    show_chat_sidebar(game_id, my_name) 
    
    st.title("🚛 Simulateur de Transport Multijoueur")
    st.caption(f"Partie ID: **{game_id}** | Joueur: **{my_name}**")
    
    # Vérication de fin de partie
    active_players = [p for p in st.session_state.players if p['active'] or p.get('can_recover')]
    
    if len([p for p in active_players if not p['is_human']]) == 0 or st.session_state.turn > 20: 
        st.error("FIN DE LA SIMULATION : La partie est terminée.")
        show_final_results()
        if st.button("Recommencer la configuration"):
            st.session_state.clear()
            st.rerun()
        return

    st.header(f"Tour actuel : {st.session_state.turn}")
    st.info(f"Événement du tour : **{st.session_state.current_event['name']}** - *{st.session_state.current_event['text']}*")
    
    # Affichage des métriques clés
    cols = st.columns(3)
    cols[0].metric("Tendance Marché", f"{st.session_state.market_trend:.2f}")
    cols[1].metric("Colis en attente (Backlog)", sum(st.session_state.backlog_packages.values()))
    cols[2].metric("Total Entreprises Actives", len(active_players))
    
    # --- Formulaire d'Actions du Joueur Connecté ---
    
    human_players = [p for p in st.session_state.players if p["is_human"]]
    player_human = next((p for p in human_players if p["name"] == my_name), None)
        
    is_ready = st.session_state.players_ready.get(my_name, False)
    
    st.divider()
    
    with st.container(border=True):
        st.subheader(f"Vos actions : **{my_name}**")
        
        if is_ready:
            st.success("✅ Vos actions sont **validées et en attente** des autres joueurs.")
        
        # Formulaire
        human_actions = get_human_actions_form(player_human, disabled=is_ready or not player_human["active"])
        
        if st.button(f"☑️ Valider les actions pour ce tour", disabled=is_ready, type="primary"):
            
            # Enregistrer l'action dans la session locale et marquer comme prêt
            # (Utilisation du deepcopy pour éviter les problèmes de modification d'objet dans Streamlit)
            if 'actions_this_turn' not in st.session_state: st.session_state.actions_this_turn = {}
            st.session_state.actions_this_turn[my_name] = deepcopy(human_actions)
            st.session_state.players_ready[my_name] = True
            
            # Sauvegarder immédiatement l'état local dans la BDD pour que les autres le voient
            save_game_state_to_db(game_id, st.session_state)
            
            st.success(f"Actions de {my_name} enregistrées. En attente des autres joueurs...")
            st.rerun()
            
    # --- Bloc d'Avancement du Tour ---
    st.divider()
    with st.container(border=True):
        st.subheader("Avancement du Tour")
        
        ready_count = sum(st.session_state.players_ready.values())
        total_human = len(human_players)
        
        st.markdown(f"**{ready_count}/{total_human}** joueurs humains ont validé leurs actions.")
        
        if ready_count == total_human:
            st.success("TOUS LES JOUEURS SONT PRÊTS. Le tour peut être exécuté.")
            if st.button("▶️ Exécuter le Prochain Tour", type="primary"):
                all_actions = st.session_state.actions_this_turn
                run_next_turn(all_actions)
                st.rerun()
        else:
            st.info(f"Joueurs en attente: {', '.join([p['name'] for p in human_players if not st.session_state.players_ready.get(p['name'])])}")
            
    st.divider()
    
    # --- Affichage des Résultats du Dernier Tour ---
    
    with st.expander("📊 Résumé du Dernier Tour et Statut Actuel", expanded=True):
        st.subheader("Statut Financier et Opérationnel")
        data = []
        for p in st.session_state.players:
             status = "ACTIF" if p["active"] else ("FAILLITE (Vendre Actifs)" if p.get("can_recover") else "LIQUIDÉE")
             data.append({
                 "Entreprise": p["name"],
                 "Statut": status,
                 "Trésorerie": f"{p['money']:,.0f} €",
                 "Dette": f"{p['loan']:,.0f} €",
                 "Réputation": f"{p['reputation']:.2f}",
                 "Capacité Totale": calculate_player_capacity(p),
                 "Revenus (T-{st.session_state.turn-1})": f"{p.get('income', 0):,.0f} €",
                 "Dépenses (T-{st.session_state.turn-1})": f"{p.get('expenses', 0):,.0f} €",
                 "Histoire du Tour": "; ".join(p.get('history', ['N/A']))
             })
        
        df = pd.DataFrame(data).set_index("Entreprise")
        st.dataframe(df, use_container_width=True)

        st.subheader("Total des Colis Livrés (Cumul)")
        show_delivery_summary(st.session_state.players)

if __name__ == "__main__":
    main()