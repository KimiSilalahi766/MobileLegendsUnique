"""
Mobile Legends Draft Pick System
Advanced draft advisor with counter recommendations and comprehensive hero database
"""

import json
import logging
from typing import Dict, List, Tuple, Optional

class MobileLegendsDraftSystem:
    def __init__(self):
        self.heroes = self._get_complete_hero_database()
        self.counter_matrix = self._build_counter_matrix()
        self.reset_draft()
        
    def reset_draft(self):
        """Reset draft state for a new session"""
        self.current_phase = "setup"  # setup, ban, pick, complete
        self.current_turn = "team"    # team or enemy
        self.rank = "Mythic"
        self.first_ban_team = "team"
        self.banned_heroes = []
        self.team_picks = []
        self.enemy_picks = []
        self.ban_sequence = []
        self.pick_sequence = []
        self.current_step = 0
        
    def _get_complete_hero_database(self) -> Dict[str, Dict]:
        """Complete Mobile Legends hero database with 129 heroes"""
        heroes = {
            # Tank Heroes (25)
            "Akai": {"role": "Tank", "difficulty": "Easy"},
            "Atlas": {"role": "Tank", "difficulty": "Hard"},
            "Baxia": {"role": "Tank", "difficulty": "Medium"},
            "Belerick": {"role": "Tank", "difficulty": "Easy"},
            "Edith": {"role": "Tank", "difficulty": "Hard"},
            "Franco": {"role": "Tank", "difficulty": "Medium"},
            "Grock": {"role": "Tank", "difficulty": "Medium"},
            "Hylos": {"role": "Tank", "difficulty": "Easy"},
            "Johnson": {"role": "Tank", "difficulty": "Hard"},
            "Khufra": {"role": "Tank", "difficulty": "Medium"},
            "Lolita": {"role": "Tank", "difficulty": "Easy"},
            "Minotaur": {"role": "Tank", "difficulty": "Medium"},
            "Tigreal": {"role": "Tank", "difficulty": "Easy"},
            "Uranus": {"role": "Tank", "difficulty": "Medium"},
            "Gatotkaca": {"role": "Tank", "difficulty": "Medium"},
            "Hilda": {"role": "Tank", "difficulty": "Easy"},
            "Jawhead": {"role": "Tank", "difficulty": "Medium"},
            "Gloo": {"role": "Tank", "difficulty": "Hard"},
            "Barats": {"role": "Tank", "difficulty": "Medium"},
            "Fredrinn": {"role": "Tank", "difficulty": "Medium"},
            "Chip": {"role": "Tank", "difficulty": "Easy"},
            "Masha": {"role": "Tank", "difficulty": "Medium"},
            "Kaja": {"role": "Tank", "difficulty": "Medium"},
            "Carmilla": {"role": "Tank", "difficulty": "Hard"},
            "Alice": {"role": "Tank", "difficulty": "Medium"},
            
            # Fighter Heroes (30)
            "Aldous": {"role": "Fighter", "difficulty": "Medium"},
            "Alpha": {"role": "Fighter", "difficulty": "Easy"},
            "Alucard": {"role": "Fighter", "difficulty": "Easy"},
            "Argus": {"role": "Fighter", "difficulty": "Medium"},
            "Badang": {"role": "Fighter", "difficulty": "Medium"},
            "Balmond": {"role": "Fighter", "difficulty": "Easy"},
            "Bane": {"role": "Fighter", "difficulty": "Easy"},
            "Chou": {"role": "Fighter", "difficulty": "Hard"},
            "Dyrroth": {"role": "Fighter", "difficulty": "Medium"},
            "Esmeralda": {"role": "Fighter", "difficulty": "Medium"},
            "Freya": {"role": "Fighter", "difficulty": "Medium"},
            "Guinevere": {"role": "Fighter", "difficulty": "Hard"},
            "Lapu-Lapu": {"role": "Fighter", "difficulty": "Medium"},
            "Leomord": {"role": "Fighter", "difficulty": "Medium"},
            "Martis": {"role": "Fighter", "difficulty": "Medium"},
            "Minsitthar": {"role": "Fighter", "difficulty": "Medium"},
            "Paquito": {"role": "Fighter", "difficulty": "Hard"},
            "Ruby": {"role": "Fighter", "difficulty": "Medium"},
            "Silvanna": {"role": "Fighter", "difficulty": "Medium"},
            "Thamuz": {"role": "Fighter", "difficulty": "Medium"},
            "Yu Zhong": {"role": "Fighter", "difficulty": "Hard"},
            "Khaleed": {"role": "Fighter", "difficulty": "Medium"},
            "Phoveus": {"role": "Fighter", "difficulty": "Medium"},
            "Aulus": {"role": "Fighter", "difficulty": "Easy"},
            "Zilong": {"role": "Fighter", "difficulty": "Easy"},
            "Sun": {"role": "Fighter", "difficulty": "Easy"},
            "Roger": {"role": "Fighter", "difficulty": "Medium"},
            "Terizla": {"role": "Fighter", "difficulty": "Medium"},
            "X.Borg": {"role": "Fighter", "difficulty": "Hard"},
            "Arlott": {"role": "Fighter", "difficulty": "Hard"},
            
            # Assassin Heroes (25)
            "Aamon": {"role": "Assassin", "difficulty": "Hard"},
            "Benedetta": {"role": "Assassin", "difficulty": "Hard"},
            "Fanny": {"role": "Assassin", "difficulty": "Hard"},
            "Gusion": {"role": "Assassin", "difficulty": "Hard"},
            "Hanzo": {"role": "Assassin", "difficulty": "Hard"},
            "Harley": {"role": "Assassin", "difficulty": "Medium"},
            "Hayabusa": {"role": "Assassin", "difficulty": "Hard"},
            "Helcurt": {"role": "Assassin", "difficulty": "Medium"},
            "Joy": {"role": "Assassin", "difficulty": "Hard"},
            "Karina": {"role": "Assassin", "difficulty": "Easy"},
            "Lancelot": {"role": "Assassin", "difficulty": "Hard"},
            "Ling": {"role": "Assassin", "difficulty": "Hard"},
            "Natalia": {"role": "Assassin", "difficulty": "Medium"},
            "Saber": {"role": "Assassin", "difficulty": "Easy"},
            "Selena": {"role": "Assassin", "difficulty": "Hard"},
            "Yi Sun-shin": {"role": "Assassin", "difficulty": "Medium"},
            "Yin": {"role": "Assassin", "difficulty": "Medium"},
            "Nolan": {"role": "Assassin", "difficulty": "Medium"},
            "Venom": {"role": "Assassin", "difficulty": "Medium"},
            "Lesley": {"role": "Assassin", "difficulty": "Medium"},
            "Claude": {"role": "Assassin", "difficulty": "Medium"},
            "Irithel": {"role": "Assassin", "difficulty": "Medium"},
            "Moskov": {"role": "Assassin", "difficulty": "Medium"},
            "Popol and Kupa": {"role": "Assassin", "difficulty": "Medium"},
            "Suyou": {"role": "Assassin", "difficulty": "Hard"},
            
            # Mage Heroes (30)
            "Aurora": {"role": "Mage", "difficulty": "Easy"},
            "Cecilion": {"role": "Mage", "difficulty": "Medium"},
            "Chang'e": {"role": "Mage", "difficulty": "Easy"},
            "Cyclops": {"role": "Mage", "difficulty": "Easy"},
            "Eudora": {"role": "Mage", "difficulty": "Easy"},
            "Faramis": {"role": "Mage", "difficulty": "Medium"},
            "Gord": {"role": "Mage", "difficulty": "Easy"},
            "Harith": {"role": "Mage", "difficulty": "Medium"},
            "Kagura": {"role": "Mage", "difficulty": "Hard"},
            "Kadita": {"role": "Mage", "difficulty": "Medium"},
            "Kimmy": {"role": "Mage", "difficulty": "Medium"},
            "Luo Yi": {"role": "Mage", "difficulty": "Hard"},
            "Lunox": {"role": "Mage", "difficulty": "Hard"},
            "Lylia": {"role": "Mage", "difficulty": "Medium"},
            "Nana": {"role": "Mage", "difficulty": "Easy"},
            "Odette": {"role": "Mage", "difficulty": "Easy"},
            "Pharsa": {"role": "Mage", "difficulty": "Medium"},
            "Valir": {"role": "Mage", "difficulty": "Medium"},
            "Vale": {"role": "Mage", "difficulty": "Medium"},
            "Vexana": {"role": "Mage", "difficulty": "Medium"},
            "Xavier": {"role": "Mage", "difficulty": "Hard"},
            "Yve": {"role": "Mage", "difficulty": "Medium"},
            "Zhask": {"role": "Mage", "difficulty": "Medium"},
            "Valentina": {"role": "Mage", "difficulty": "Hard"},
            "Novaria": {"role": "Mage", "difficulty": "Medium"},
            "Julian": {"role": "Mage", "difficulty": "Hard"},
            "Lukas": {"role": "Mage", "difficulty": "Medium"},
            "Melissa": {"role": "Mage", "difficulty": "Medium"},
            "Natan": {"role": "Mage", "difficulty": "Medium"},
            "Wanwan": {"role": "Mage", "difficulty": "Medium"},
            
            # Marksman Heroes (19)
            "Beatrix": {"role": "Marksman", "difficulty": "Hard"},
            "Bruno": {"role": "Marksman", "difficulty": "Medium"},
            "Clint": {"role": "Marksman", "difficulty": "Medium"},
            "Granger": {"role": "Marksman", "difficulty": "Medium"},
            "Hanabi": {"role": "Marksman", "difficulty": "Easy"},
            "Karrie": {"role": "Marksman", "difficulty": "Medium"},
            "Layla": {"role": "Marksman", "difficulty": "Easy"},
            "Miya": {"role": "Marksman", "difficulty": "Easy"},
            "Brody": {"role": "Marksman", "difficulty": "Medium"},
            "Ixia": {"role": "Marksman", "difficulty": "Medium"},
        }
        
        return heroes
    
    def _build_counter_matrix(self) -> Dict[str, List[str]]:
        """Build comprehensive counter matrix for meta heroes"""
        counter_matrix = {
            # Meta Assassins
            "Fanny": ["Khufra", "Franco", "Natalia", "Saber", "Kaja", "Ruby", "Jawhead"],
            "Ling": ["Khufra", "Franco", "Saber", "Selena", "Natalia", "Kaja", "Jawhead"],
            "Lancelot": ["Saber", "Ruby", "Franco", "Khufra", "Natalia", "Kaja"],
            "Gusion": ["Saber", "Franco", "Natalia", "Ruby", "Khufra", "Kaja"],
            "Hayabusa": ["Saber", "Franco", "Ruby", "Khufra", "Natalia"],
            "Benedetta": ["Franco", "Ruby", "Khufra", "Saber", "Natalia"],
            "Aamon": ["Franco", "Ruby", "Saber", "Khufra", "Selena"],
            
            # Meta Mages
            "Kagura": ["Lancelot", "Gusion", "Hayabusa", "Fanny", "Harley", "Franco"],
            "Xavier": ["Lancelot", "Gusion", "Fanny", "Hayabusa", "Franco", "Khufra"],
            "Valentina": ["Franco", "Khufra", "Lancelot", "Gusion", "Saber"],
            "Lunox": ["Lancelot", "Gusion", "Franco", "Khufra", "Hayabusa"],
            "Yve": ["Lancelot", "Fanny", "Gusion", "Franco", "Hayabusa"],
            "Pharsa": ["Fanny", "Lancelot", "Gusion", "Franco", "Hayabusa"],
            "Cecilion": ["Lancelot", "Gusion", "Franco", "Fanny", "Hayabusa"],
            
            # Meta Marksman
            "Beatrix": ["Fanny", "Lancelot", "Franco", "Khufra", "Gusion"],
            "Granger": ["Fanny", "Lancelot", "Franco", "Khufra", "Gusion"],
            "Brody": ["Fanny", "Lancelot", "Franco", "Khufra", "Gusion"],
            "Karrie": ["Fanny", "Lancelot", "Franco", "Khufra", "Gusion"],
            "Melissa": ["Fanny", "Lancelot", "Franco", "Khufra", "Gusion"],
            
            # Meta Fighters
            "Paquito": ["Ruby", "Chou", "Franco", "Khufra", "Yu Zhong"],
            "Yu Zhong": ["Karrie", "Melissa", "Ruby", "Franco", "Khufra"],
            "Chou": ["Ruby", "Franco", "Khufra", "Yu Zhong", "Paquito"],
            "Esmeralda": ["Karrie", "Melissa", "Ruby", "Franco", "Khufra"],
            "Silvanna": ["Ruby", "Chou", "Franco", "Khufra", "Yu Zhong"],
            
            # Meta Tanks
            "Khufra": ["Karrie", "Ruby", "Melissa", "Granger", "Chou"],
            "Franco": ["Ruby", "Chou", "Karrie", "Melissa", "Granger"],
            "Atlas": ["Ruby", "Chou", "Karrie", "Melissa", "Granger"],
            "Tigreal": ["Ruby", "Chou", "Karrie", "Melissa", "Granger"],
            "Johnson": ["Ruby", "Chou", "Karrie", "Melissa", "Granger"],
            
            # Support/Others
            "Estes": ["Lancelot", "Gusion", "Fanny", "Franco", "Khufra"],
            "Rafaela": ["Lancelot", "Gusion", "Fanny", "Franco", "Khufra"],
            "Mathilda": ["Lancelot", "Gusion", "Franco", "Khufra", "Saber"],
        }
        
        return counter_matrix
    
    def get_ban_sequence(self, rank: str, first_team: str) -> List[str]:
        """Get ban sequence based on rank and first ban team"""
        sequences = {
            "Epic": ["team", "enemy", "team", "enemy", "team", "enemy"] if first_team == "team" 
                    else ["enemy", "team", "enemy", "team", "enemy", "team"],
            "Legend": ["team", "enemy", "team", "enemy", "team", "enemy", "team", "enemy"] if first_team == "team"
                     else ["enemy", "team", "enemy", "team", "enemy", "team", "enemy", "team"],
            "Mythic": ["team", "enemy", "team", "enemy", "team", "enemy", "team", "enemy", "team", "enemy"] if first_team == "team"
                      else ["enemy", "team", "enemy", "team", "enemy", "team", "enemy", "team", "enemy", "team"]
        }
        return sequences.get(rank, sequences["Mythic"])
    
    def get_pick_sequence(self, first_team: str) -> List[str]:
        """Get pick sequence (always same regardless of rank)"""
        if first_team == "team":
            return ["team", "enemy", "enemy", "team", "team", "enemy", "enemy", "team", "team", "enemy"]
        else:
            return ["enemy", "team", "team", "enemy", "enemy", "team", "team", "enemy", "enemy", "team"]
    
    def start_draft(self, rank: str, first_ban_team: str):
        """Initialize draft with given parameters"""
        self.rank = rank
        self.first_ban_team = first_ban_team
        self.ban_sequence = self.get_ban_sequence(rank, first_ban_team)
        self.pick_sequence = self.get_pick_sequence(first_ban_team)
        self.current_phase = "ban"
        self.current_turn = self.ban_sequence[0]
        self.current_step = 0
        
        logging.info(f"Draft started: {rank} rank, {first_ban_team} first ban")
    
    def process_ban(self, hero_name: str) -> bool:
        """Process ban action"""
        if self.current_phase != "ban" or hero_name in self.banned_heroes:
            return False
        
        self.banned_heroes.append(hero_name)
        self.current_step += 1
        
        # Check if ban phase is complete
        if self.current_step >= len(self.ban_sequence):
            self.current_phase = "pick"
            self.current_step = 0
            self.current_turn = self.pick_sequence[0]
        else:
            self.current_turn = self.ban_sequence[self.current_step]
        
        logging.info(f"Banned {hero_name}, step {self.current_step}")
        return True
    
    def process_pick(self, hero_name: str) -> bool:
        """Process pick action"""
        if self.current_phase != "pick" or hero_name in self.banned_heroes:
            return False
        
        if hero_name in self.team_picks or hero_name in self.enemy_picks:
            return False
        
        if self.current_turn == "team":
            self.team_picks.append(hero_name)
        else:
            self.enemy_picks.append(hero_name)
        
        self.current_step += 1
        
        # Check if pick phase is complete
        if self.current_step >= len(self.pick_sequence):
            self.current_phase = "complete"
        else:
            self.current_turn = self.pick_sequence[self.current_step]
        
        logging.info(f"Picked {hero_name} for {self.current_turn}, step {self.current_step}")
        return True
    
    def get_recommendations(self) -> List[Dict[str, str]]:
        """Get AI counter recommendations for current situation"""
        if self.current_phase != "pick" or self.current_turn != "team":
            return []
        
        recommendations = []
        
        # Get counters for enemy picks
        for enemy_hero in self.enemy_picks:
            if enemy_hero in self.counter_matrix:
                for counter in self.counter_matrix[enemy_hero]:
                    if (counter not in self.banned_heroes and 
                        counter not in self.team_picks and 
                        counter not in self.enemy_picks and
                        counter in self.heroes):
                        
                        recommendations.append({
                            "hero": counter,
                            "target": enemy_hero,
                            "role": self.heroes[counter]["role"],
                            "reason": f"Counters {enemy_hero}"
                        })
        
        # Remove duplicates and limit to top 8
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec["hero"] not in seen:
                seen.add(rec["hero"])
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= 8:
                    break
        
        return unique_recommendations
    
    def search_heroes(self, query: str) -> List[str]:
        """Search heroes by name"""
        if not query or len(query.strip()) < 1:
            # Return first 9 available heroes if no query
            available_heroes = []
            for hero_name in self.heroes.keys():
                if (hero_name not in self.banned_heroes and
                    hero_name not in self.team_picks and
                    hero_name not in self.enemy_picks):
                    available_heroes.append(hero_name)
                if len(available_heroes) >= 9:
                    break
            return available_heroes
        
        query = query.lower().strip()
        matches = []
        
        # First: exact matches at start of name
        for hero_name in self.heroes.keys():
            if (hero_name.lower().startswith(query) and 
                hero_name not in self.banned_heroes and
                hero_name not in self.team_picks and
                hero_name not in self.enemy_picks):
                matches.append(hero_name)
        
        # Second: partial matches anywhere in name
        for hero_name in self.heroes.keys():
            if (query in hero_name.lower() and 
                hero_name not in matches and
                hero_name not in self.banned_heroes and
                hero_name not in self.team_picks and
                hero_name not in self.enemy_picks):
                matches.append(hero_name)
        
        logging.info(f"Hero search for '{query}': found {len(matches)} matches")
        return matches[:9]  # Limit to 9 results
    
    def get_draft_state(self) -> Dict:
        """Get current draft state"""
        return {
            "phase": self.current_phase,
            "turn": self.current_turn,
            "rank": self.rank,
            "step": self.current_step,
            "banned_heroes": self.banned_heroes,
            "team_picks": self.team_picks,
            "enemy_picks": self.enemy_picks,
            "ban_sequence": self.ban_sequence,
            "pick_sequence": self.pick_sequence
        }
    
    def analyze_team_composition(self) -> Dict:
        """Analyze final team composition with win rate prediction"""
        if len(self.team_picks) < 5:
            return {}
        
        roles = {}
        for hero in self.team_picks:
            role = self.heroes.get(hero, {}).get("role", "Unknown")
            roles[role] = roles.get(role, 0) + 1
        
        # Calculate composition score
        ideal_roles = {"Tank": 1, "Fighter": 1, "Assassin": 1, "Mage": 1, "Marksman": 1}
        composition_score = 0
        
        for role, ideal_count in ideal_roles.items():
            actual_count = roles.get(role, 0)
            if actual_count == ideal_count:
                composition_score += 20
            elif actual_count > 0:
                composition_score += 10
        
        # Calculate win rate prediction based on multiple factors
        win_rate_analysis = self._calculate_win_rate_prediction(roles)
        
        return {
            "team_picks": self.team_picks,
            "enemy_picks": self.enemy_picks,
            "roles": roles,
            "composition_score": composition_score,
            "win_rate_prediction": win_rate_analysis["win_rate"],
            "win_rate_factors": win_rate_analysis["factors"],
            "analysis": self._get_composition_analysis(roles),
            "detailed_analysis": win_rate_analysis["detailed_analysis"]
        }
    
    def _get_composition_analysis(self, roles: Dict) -> str:
        """Generate composition analysis text"""
        analysis = []
        
        if roles.get("Tank", 0) == 0:
            analysis.append("Missing Tank - Team may lack initiation and protection")
        if roles.get("Marksman", 0) == 0:
            analysis.append("Missing Marksman - Team may lack sustained damage")
        if roles.get("Mage", 0) == 0:
            analysis.append("Missing Mage - Team may lack magic damage")
        
        if roles.get("Tank", 0) > 1:
            analysis.append("Multiple Tanks - Good for team fights but may lack damage")
        if roles.get("Assassin", 0) > 1:
            analysis.append("Multiple Assassins - High burst potential but risky")
        
        if not analysis:
            analysis.append("Balanced composition - Good mix of roles")
        
        return " | ".join(analysis)
    
    def _calculate_win_rate_prediction(self, team_roles: Dict) -> Dict:
        """Calculate win rate prediction based on multiple factors"""
        base_win_rate = 50.0  # Starting point
        factors = []
        detailed_analysis = []
        
        # 1. Composition Balance Analysis (±15%)
        composition_bonus = self._analyze_composition_balance(team_roles)
        base_win_rate += composition_bonus["bonus"]
        factors.append(f"Komposisi Tim: {composition_bonus['bonus']:+.1f}%")
        detailed_analysis.extend(composition_bonus["reasons"])
        
        # 2. Counter Pick Analysis (±20%)
        counter_analysis = self._analyze_counter_picks()
        base_win_rate += counter_analysis["bonus"]
        factors.append(f"Counter Picks: {counter_analysis['bonus']:+.1f}%")
        detailed_analysis.extend(counter_analysis["reasons"])
        
        # 3. Meta Strength Analysis (±10%)
        meta_analysis = self._analyze_meta_strength()
        base_win_rate += meta_analysis["bonus"]
        factors.append(f"Meta Strength: {meta_analysis['bonus']:+.1f}%")
        detailed_analysis.extend(meta_analysis["reasons"])
        
        # 4. Synergy Analysis (±8%)
        synergy_analysis = self._analyze_team_synergy()
        base_win_rate += synergy_analysis["bonus"]
        factors.append(f"Team Synergy: {synergy_analysis['bonus']:+.1f}%")
        detailed_analysis.extend(synergy_analysis["reasons"])
        
        # 5. Early/Late Game Balance (±7%)
        game_phase_analysis = self._analyze_game_phase_strength()
        base_win_rate += game_phase_analysis["bonus"]
        factors.append(f"Game Phase: {game_phase_analysis['bonus']:+.1f}%")
        detailed_analysis.extend(game_phase_analysis["reasons"])
        
        # Clamp win rate between 15% and 85%
        final_win_rate = max(15.0, min(85.0, base_win_rate))
        
        return {
            "win_rate": round(final_win_rate, 1),
            "factors": factors,
            "detailed_analysis": detailed_analysis
        }
    
    def _analyze_composition_balance(self, roles: Dict) -> Dict:
        """Analyze team composition balance"""
        bonus = 0.0
        reasons = []
        
        # Ideal composition: 1 Tank, 1 Fighter, 1 Assassin, 1 Mage, 1 Marksman
        ideal_roles = {"Tank": 1, "Fighter": 1, "Assassin": 1, "Mage": 1, "Marksman": 1}
        
        # Perfect composition bonus
        if all(roles.get(role, 0) == count for role, count in ideal_roles.items()):
            bonus += 12.0
            reasons.append("✓ Komposisi ideal (1-1-1-1-1) memberikan keseimbangan sempurna")
        else:
            # Analyze specific role issues
            if roles.get("Tank", 0) == 0:
                bonus -= 10.0
                reasons.append("✗ Tanpa Tank: Sulit initiate dan protect carry (-10%)")
            elif roles.get("Tank", 0) > 1:
                bonus -= 5.0
                reasons.append("✗ Terlalu banyak Tank: Kurang damage output (-5%)")
            
            if roles.get("Marksman", 0) == 0:
                bonus -= 8.0
                reasons.append("✗ Tanpa Marksman: Kurang sustained damage late game (-8%)")
            elif roles.get("Marksman", 0) > 1:
                bonus -= 4.0
                reasons.append("✗ Terlalu banyak Marksman: Rentan di early game (-4%)")
            
            if roles.get("Mage", 0) == 0:
                bonus -= 6.0
                reasons.append("✗ Tanpa Mage: Kurang magic damage dan burst (-6%)")
            
            # Multiple assassins risk
            if roles.get("Assassin", 0) > 1:
                bonus -= 7.0
                reasons.append("✗ Terlalu banyak Assassin: High risk, sulit team fight (-7%)")
            
            # Decent alternatives
            if roles.get("Tank", 0) == 1 and roles.get("Marksman", 0) == 1:
                bonus += 6.0
                reasons.append("✓ Tank + Marksman core: Foundation solid (+6%)")
        
        return {"bonus": bonus, "reasons": reasons}
    
    def _analyze_counter_picks(self) -> Dict:
        """Analyze counter pick effectiveness"""
        bonus = 0.0
        reasons = []
        
        if len(self.enemy_picks) < 5:
            return {"bonus": 0.0, "reasons": ["Counter analysis memerlukan draft lengkap"]}
        
        countered_enemies = 0
        counter_details = []
        
        for team_hero in self.team_picks:
            counters = self.counter_matrix.get(team_hero, [])
            for enemy_hero in self.enemy_picks:
                if enemy_hero in counters:
                    countered_enemies += 1
                    counter_details.append(f"{team_hero} vs {enemy_hero}")
        
        # Counter effectiveness scoring
        if countered_enemies >= 4:
            bonus += 15.0
            reasons.append(f"✓ Excellent counters: {countered_enemies} matchups menguntungkan (+15%)")
        elif countered_enemies >= 3:
            bonus += 10.0
            reasons.append(f"✓ Good counters: {countered_enemies} matchups menguntungkan (+10%)")
        elif countered_enemies >= 2:
            bonus += 5.0
            reasons.append(f"✓ Decent counters: {countered_enemies} matchups menguntungkan (+5%)")
        elif countered_enemies == 1:
            bonus += 2.0
            reasons.append(f"~ Minimal counters: {countered_enemies} matchup menguntungkan (+2%)")
        else:
            bonus -= 8.0
            reasons.append("✗ No effective counters: Tim tidak mengcounter musuh (-8%)")
        
        # Enemy countering us (penalty)
        our_heroes_countered = 0
        for enemy_hero in self.enemy_picks:
            enemy_counters = self.counter_matrix.get(enemy_hero, [])
            for team_hero in self.team_picks:
                if team_hero in enemy_counters:
                    our_heroes_countered += 1
        
        if our_heroes_countered >= 3:
            bonus -= 12.0
            reasons.append(f"✗ Heavily countered: {our_heroes_countered} hero kami di-counter (-12%)")
        elif our_heroes_countered >= 2:
            bonus -= 6.0
            reasons.append(f"✗ Partially countered: {our_heroes_countered} hero kami di-counter (-6%)")
        
        if counter_details:
            reasons.append(f"Matchups kunci: {', '.join(counter_details[:3])}")
        
        return {"bonus": bonus, "reasons": reasons}
    
    def _analyze_meta_strength(self) -> Dict:
        """Analyze current meta strength of picked heroes"""
        bonus = 0.0
        reasons = []
        
        # Define meta tiers (simplified)
        s_tier_heroes = ["Fanny", "Lancelot", "Gusion", "Harley", "Claude", "Granger", "Ling", "Chou", "Khufra", "Atlas"]
        a_tier_heroes = ["Hayabusa", "Hanzo", "Karrie", "Bruno", "Kimmy", "Esmeralda", "X.Borg", "Grock", "Johnson"]
        
        s_tier_count = sum(1 for hero in self.team_picks if hero in s_tier_heroes)
        a_tier_count = sum(1 for hero in self.team_picks if hero in a_tier_heroes)
        
        # Meta strength scoring
        if s_tier_count >= 3:
            bonus += 8.0
            reasons.append(f"✓ {s_tier_count} S-tier heroes: Meta dominance (+8%)")
        elif s_tier_count >= 2:
            bonus += 5.0
            reasons.append(f"✓ {s_tier_count} S-tier heroes: Strong meta picks (+5%)")
        elif s_tier_count >= 1:
            bonus += 2.0
            reasons.append(f"✓ {s_tier_count} S-tier hero: Decent meta presence (+2%)")
        
        if a_tier_count >= 2:
            bonus += 3.0
            reasons.append(f"✓ {a_tier_count} A-tier heroes: Solid meta choices (+3%)")
        
        # Off-meta penalty
        meta_heroes = len([h for h in self.team_picks if h in s_tier_heroes + a_tier_heroes])
        if meta_heroes <= 1:
            bonus -= 6.0
            reasons.append("✗ Mostly off-meta picks: Dapat tertinggal power level (-6%)")
        
        return {"bonus": bonus, "reasons": reasons}
    
    def _analyze_team_synergy(self) -> Dict:
        """Analyze team synergy potential"""
        bonus = 0.0
        reasons = []
        
        # Define synergy combinations
        combo_teams = {
            "Wombo Combo": ["Atlas", "Vale", "Pharsa", "Kagura", "Aurora"],
            "Protect the Carry": ["Johnson", "Lolita", "Claude", "Karrie", "Granger"], 
            "Dive Comp": ["Khufra", "Chou", "Fanny", "Gusion", "Ling"],
            "Poke Comp": ["Chang'e", "Pharsa", "Cecilion", "Kimmy", "Valir"],
            "Split Push": ["Hayabusa", "Fanny", "Sun", "Zilong", "Aldous"]
        }
        
        best_synergy = 0
        synergy_type = ""
        
        for comp_name, heroes in combo_teams.items():
            overlap = len(set(self.team_picks) & set(heroes))
            if overlap > best_synergy:
                best_synergy = overlap
                synergy_type = comp_name
        
        # Synergy scoring
        if best_synergy >= 4:
            bonus += 6.0
            reasons.append(f"✓ Excellent synergy: {synergy_type} composition (+6%)")
        elif best_synergy >= 3:
            bonus += 4.0
            reasons.append(f"✓ Good synergy: {synergy_type} elements (+4%)")
        elif best_synergy >= 2:
            bonus += 2.0
            reasons.append(f"✓ Some synergy: {synergy_type} potential (+2%)")
        else:
            bonus -= 3.0
            reasons.append("✗ Limited synergy: Heroes tidak saling mendukung (-3%)")
        
        return {"bonus": bonus, "reasons": reasons}
    
    def _analyze_game_phase_strength(self) -> Dict:
        """Analyze early vs late game balance"""
        bonus = 0.0
        reasons = []
        
        # Classify heroes by game phase strength
        early_game_heroes = ["Tigreal", "Akai", "Johnson", "Bruno", "Clint", "Chou", "Leomord"]
        late_game_heroes = ["Aldous", "Claude", "Karrie", "Hanabi", "Cecilion", "Zhask", "Angela"]
        
        early_count = sum(1 for hero in self.team_picks if hero in early_game_heroes)
        late_count = sum(1 for hero in self.team_picks if hero in late_game_heroes)
        
        # Balance scoring
        if early_count >= 2 and late_count >= 2:
            bonus += 5.0
            reasons.append("✓ Balanced game phases: Strong di early dan late game (+5%)")
        elif early_count >= 3:
            bonus += 2.0
            reasons.append("✓ Early game focused: Harus close game cepat (+2%)")
        elif late_count >= 3:
            bonus += 1.0
            reasons.append("~ Late game focused: Harus survive early game (+1%)")
        else:
            bonus -= 2.0
            reasons.append("✗ Unclear game plan: Tidak ada fase dominan (-2%)")
        
        return {"bonus": bonus, "reasons": reasons}
    
    def export_draft_result(self) -> str:
        """Export draft result as JSON string"""
        result = {
            "rank": self.rank,
            "first_ban_team": self.first_ban_team,
            "banned_heroes": self.banned_heroes,
            "team_picks": self.team_picks,
            "enemy_picks": self.enemy_picks,
            "analysis": self.analyze_team_composition()
        }
        return json.dumps(result, indent=2)