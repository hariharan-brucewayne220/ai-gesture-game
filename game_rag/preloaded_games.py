#!/usr/bin/env python3
"""
Pre-loaded Game Guides Database
Curated game guides for instant RAG system use
"""

import os
import sys
from typing import Dict, List
sys.path.append('.')

from game_rag.vector_store import GameVectorStore
from game_rag.query_engine import GameRAGEngine

class PreloadedGames:
    """Pre-loaded game guides database"""
    
    def __init__(self):
        self.vector_store = GameVectorStore()
        self.rag_engine = GameRAGEngine()
        self.games_loaded = []
        
    def load_all_games(self):
        """Load all pre-curated game guides"""
        print("Loading pre-curated game guides...")
        
        # Load each game's guides
        for game_name, guides in self.get_game_guides().items():
            print(f"Loading {game_name}...")
            self.load_game_guides(game_name, guides)
            self.games_loaded.append(game_name)
        
        print(f"Successfully loaded {len(self.games_loaded)} games!")
        return self.games_loaded
    
    def load_game_guides(self, game_name: str, guides: List[Dict]):
        """Load guides for a specific game"""
        documents = []
        metadatas = []
        
        for guide in guides:
            documents.append(guide['content'])
            metadatas.append({
                'game': game_name,
                'topic': guide['topic'],
                'source': guide['source'],
                'difficulty': guide.get('difficulty', 'medium'),
                'guide_type': guide.get('type', 'general')
            })
        
        # Add to vector store
        self.vector_store.add_documents(documents, metadatas)
    
    def get_game_guides(self) -> Dict:
        """Return all pre-curated game guides"""
        return {
            "God of War": self.get_god_of_war_guides(),
            "Elden Ring": self.get_elden_ring_guides(),
            "The Witcher 3": self.get_witcher3_guides(),
            "Dark Souls 3": self.get_dark_souls_guides(),
            "Sekiro": self.get_sekiro_guides(),
            "Cyberpunk 2077": self.get_cyberpunk_guides(),
            "Assassin's Creed Valhalla": self.get_ac_valhalla_guides(),
            "Call of Duty Warzone": self.get_cod_warzone_guides(),
            "Destiny 2": self.get_destiny2_guides(),
            "Horizon Zero Dawn": self.get_horizon_guides(),
            "Ghost of Tsushima": self.get_ghost_tsushima_guides(),
            "Bloodborne": self.get_bloodborne_guides(),
            "Nioh 2": self.get_nioh2_guides(),
            "Monster Hunter World": self.get_mhw_guides(),
            "Red Dead Redemption 2": self.get_rdr2_guides(),
            "Grand Theft Auto V": self.get_gtav_guides(),
            "Minecraft": self.get_minecraft_guides(),
            "Fortnite": self.get_fortnite_guides(),
            "Apex Legends": self.get_apex_guides(),
            "Valorant": self.get_valorant_guides()
        }
    
    def get_god_of_war_guides(self) -> List[Dict]:
        """God of War comprehensive guides"""
        return [
            {
                'topic': 'boss_battles',
                'source': 'curated_boss_guide',
                'difficulty': 'hard',
                'type': 'combat',
                'content': """
                God of War Boss Battle Strategies

                BALDUR (Final Boss):
                Phase 1: Baldur is invulnerable - focus on dodging and learning patterns
                - Use shield to block his heavy strikes
                - Dodge roll when he does his charge attack
                - Don't waste rage meter, save for later phases
                
                Phase 2: Vulnerable Baldur
                - Heavy axe throws for maximum damage
                - Use Atreus arrows to stun him
                - Combo: Light-Light-Heavy for optimal damage
                - When he's stunned, use heavy runic attacks
                
                Phase 3: Enraged Baldur
                - More aggressive attacks and grabs
                - Use Spartan Rage when health is low
                - Focus on perfect dodges to avoid unblockable attacks
                - Finish with brutal takedown prompts

                VALKYRIES:
                General Strategy:
                - Learn each Valkyrie's unique attacks
                - Use resurrection stones for tough fights
                - Upgrade armor to level 7+ before attempting
                - Practice dodging unblockable attacks (red indicators)
                
                Sigrun (Valkyrie Queen):
                - Hardest boss in the game
                - Combines attacks from all other Valkyries
                - Use Blessing of the Frost runic attack
                - Equip Deadly Mist armor set
                - Be patient - fight can take 10+ minutes
                
                TROLLS:
                - Target the weak spot on their back
                - Use axe throw to hit distant weak spots
                - Dodge their ground slam attacks
                - Use environmental hazards when available
                
                ANCIENT ENEMIES:
                - High health pools, use heavy attacks
                - Target glowing weak points
                - Use Atreus' shock arrows for crowd control
                - Runic attacks deal percentage-based damage
                """
            },
            {
                'topic': 'combat_system',
                'source': 'curated_combat_guide',
                'difficulty': 'medium',
                'type': 'combat',
                'content': """
                God of War Combat Mastery Guide

                LEVIATHAN AXE COMBAT:
                Basic Attacks:
                - Light Attack (R1): Fast combo starter
                - Heavy Attack (R2): Slow but powerful
                - Charged Heavy (Hold R2): Unblockable crusher
                
                Axe Throwing:
                - Aimed Throw (R2): Precise ranged attack
                - Quick Throw (R1 while aiming): Faster but less damage
                - Recall (Triangle): Brings axe back, can hit enemies on return
                
                Best Combos:
                - R1-R1-R2: Standard combo ender
                - R1-R1-R1-R2: Extended combo for crowds
                - Throw-Recall-R2: Ranged to melee transition
                
                BARE-HANDED COMBAT:
                - Faster attack speed than axe
                - Better for crowd control
                - Builds rage meter quicker
                - Use when axe is thrown and enemies are close
                
                SHIELD TECHNIQUES:
                - Block (L1): Reduces damage from most attacks
                - Parry (L1 at perfect timing): Staggers enemies and opens counterattack
                - Shield Bash (L1+R1): Interrupts enemy attacks
                - Perfect timing parry creates slow-motion window
                
                ATREUS COMBAT SUPPORT:
                - Arrow Command (Square): Stuns enemies
                - Runic Summons: Powerful area attacks
                - Upgrade arrows: Shock, Burn, Talon types
                - Atreus can grab ledges and reach high areas
                
                SPARTAN RAGE:
                - Activate when rage meter is full (L3+R3)
                - Increases damage and provides health regeneration
                - Use when overwhelmed or against tough enemies
                - Punches break enemy guard instantly
                - Duration increases with rage meter upgrades
                
                RUNIC ATTACKS:
                Light Runic Attacks:
                - Hel's Touch: Burn damage over time
                - Frost Giant's Frenzy: Multiple hits with freeze
                - Blessing of the Frost: High damage, good for bosses
                
                Heavy Runic Attacks:
                - Glaive Storm: Great for crowds
                - Wrath of the Frost Ancient: Massive single target damage
                - Ivaldi's Anvil: Launches enemies into air
                
                ADVANCED TECHNIQUES:
                - Animation Canceling: Use dodge to cancel attack animations
                - Rage Canceling: Enter rage to cancel recovery frames
                - Axe Juggling: Throw axe, fight bare-handed, recall for combo
                - Environmental Kills: Use walls and hazards for instant kills
                """
            },
            {
                'topic': 'exploration',
                'source': 'curated_exploration_guide',
                'difficulty': 'easy',
                'type': 'collectibles',
                'content': """
                God of War Exploration and Collectibles Guide

                NORNIR CHESTS:
                - Require solving rune puzzles to open
                - Look for 3 matching runes in the area
                - Some runes are on spinning mechanisms
                - Use Atreus' arrows to activate distant runes
                - Use axe throws to spin rune mechanisms
                - Contain Idunn Apples (health) or Horns of Blood Mead (rage)
                
                ODIN'S RAVENS (51 total):
                - Green-glowing ravens perched around the world
                - Throw axe or use arrows to destroy them
                - Some ravens are hidden behind breakable walls
                - Some require specific abilities to reach
                - Destroying all ravens unlocks unique rewards
                - Check behind waterfalls and in hidden areas
                
                ARTIFACTS:
                - Red orb markers indicate artifact locations
                - Provide background lore and world-building
                - Some artifacts require specific tools to reach
                - Family Heirloom artifacts tell personal stories
                - Jotnar Shrines provide giant mythology
                
                REALM TEARS:
                - Purple rifts that spawn challenging enemies
                - Defeat all enemies to close the tear permanently
                - Reward rare crafting materials and resources
                - Higher level tears require better equipment
                - Some tears have specific enemy types (Travelers, Revenants)
                
                LEGENDARY CHESTS:
                - Contain powerful runic attacks and rare materials
                - Often require puzzle solving to access
                - Some are locked behind story progression
                - Look for golden chest shimmer effect
                - Essential for upgrading equipment
                
                HIDDEN AREAS:
                - Breakable walls marked by different textures
                - Secret passages behind waterfalls
                - Climbable areas not obvious from ground level
                - Some areas require specific story abilities
                - Alfheim has many vertical exploration opportunities
                
                EXPLORATION TIPS:
                - Always check behind you when entering new areas
                - Use Atreus to reach high switches and levers
                - Spartan Rage can break certain barriers
                - Some areas become accessible only after story progression
                - Return to previous areas with new abilities
                - Listen for audio cues indicating hidden secrets
                """
            },
            {
                'topic': 'character_progression',
                'source': 'curated_progression_guide',
                'difficulty': 'medium',
                'type': 'progression',
                'content': """
                God of War Character Progression Guide

                KRATOS SKILL TREES:
                Axe Skills:
                - Executioner's Cleave: Powerful overhead attack
                - Whirlwind Sweep: 360-degree crowd control
                - Firing Aim: Improved axe throwing accuracy
                - Deadly Whirlwind: Enhanced whirlwind with knockdown
                
                Shield Skills:
                - Defensive Expertise: Longer parry window
                - Stunning Parry: Parries create longer stun
                - Guardian's Resolve: Shield bash deals more damage
                - Protective Barrier: Reduced damage when blocking
                
                Unarmed Skills:
                - Bare-Handed Expertise: Faster punch combos
                - Stunning Grab: Grabs stun enemies longer
                - Rage Mode: Extended Spartan Rage duration
                - Berserker's Resolve: Rage builds faster
                
                ATREUS PROGRESSION:
                - Levels up automatically through story
                - Find Cipher Chests to unlock new abilities
                - Upgrade bow at shops using crafting materials
                - Higher levels provide better combat support
                - Learns new arrow types throughout the game
                
                ARMOR SYSTEM:
                Armor Stats:
                - Strength: Increases damage output
                - Runic: Improves runic attack damage and cooldown
                - Defense: Reduces incoming damage
                - Vitality: Increases maximum health
                - Luck: Improves XP and Hacksilver gain
                
                Best Armor Sets:
                Early Game: Traveler's Set (balanced stats)
                Mid Game: Ancient Set (high runic power)
                Late Game: Ivaldi's Mist Set (best overall stats)
                End Game: Valkyrie Sets (specialized builds)
                
                ENCHANTMENTS:
                Socket System:
                - Armor pieces can have 1-3 enchantment slots
                - Enchantments provide passive bonuses
                - Stack similar enchantments for greater effect
                - Some enchantments have set bonuses
                
                Best Enchantments:
                - Vitality: Increases health significantly
                - Strength: Boosts damage output
                - Runic: Reduces runic attack cooldowns
                - Luck: Improves resource gains
                - Cooldown: Reduces all ability cooldowns
                
                WEAPON UPGRADES:
                Leviathan Axe:
                - Collect Frozen Flames to upgrade (6 total)
                - Each upgrade increases damage and unlocks new abilities
                - Fully upgraded axe has unique visual effects
                - Upgrades also improve throwing damage
                
                Blades of Chaos:
                - Collect Chaos Flames to upgrade (5 total)
                - Obtained later in the game
                - Upgrades affect both damage and range
                - Fire damage increases with upgrades
                
                RESOURCES AND MATERIALS:
                Hacksilver: Primary currency for upgrades
                XP: Spend on skill tree abilities
                Rare Materials: Used for high-level crafting
                Aegir's Gold: Rare material for best equipment
                World Serpent Scales: Upgrade materials from boss
                
                PROGRESSION TIPS:
                - Focus on one skill tree at a time
                - Save rare materials for end-game equipment
                - Upgrade weapons before armor in early game
                - Experiment with different enchantment combinations
                - Don't ignore Atreus upgrades - they're crucial for combat
                """
            }
        ]
    
    def get_elden_ring_guides(self) -> List[Dict]:
        """Elden Ring comprehensive guides"""
        return [
            {
                'topic': 'boss_strategies',
                'source': 'curated_boss_guide',
                'difficulty': 'hard',
                'type': 'combat',
                'content': """
                Elden Ring Boss Strategy Guide

                MARGIT THE FELL OMEN:
                Phase 1:
                - Stay close to avoid his jumping attacks
                - Attack his legs to stagger him
                - Use Margit's Shackle item to bind him temporarily
                - Roll through his staff combo, don't back away
                
                Phase 2 (50% health):
                - He gains a hammer - more aggressive attacks
                - Use pillars as cover from ranged attacks
                - Summon Rogier for additional help
                - Focus on hit-and-run tactics
                
                GODRICK THE GRAFTED:
                Phase 1:
                - Attack his ankles to topple him
                - Use Spirit Summons to distract him
                - Dodge roll his axe slams
                - Stay behind him when possible
                
                Phase 2 (Dragon hand):
                - Keep distance from fire breath
                - Attack the dragon hand when he breathes fire
                - Use ranged attacks or magic
                - Nepheli Loux summon available for help
                
                RENNALA QUEEN OF THE FULL MOON:
                Phase 1:
                - Attack students with golden auras
                - This drops Rennala from the air
                - Attack her while she's down
                - Repeat until phase transition
                
                Phase 2:
                - Dodge her beam attacks by staying mobile
                - Use Torrent-like mobility
                - Attack during her long casting animations
                - Be ready for her summons (dragon, wolves)
                
                RADAHN THE STARSCOURGE:
                Pre-Fight:
                - Summon ALL available NPCs
                - Use Torrent for mobility
                - Stock up on arrows if using ranged
                
                Strategy:
                - Let summons tank while you attack from behind
                - Use hit-and-run tactics on horseback
                - Re-summon fallen NPCs during fight
                - Stay mobile to avoid his gravity arrows
                - In final phase, be ready for his meteor attack
                
                MALENIA BLADE OF MIQUELLA:
                Phase 1:
                - Extremely aggressive - dodge everything
                - She heals on hit, even through shields
                - Use frost or bleed weapons
                - Summon Mimic Tear for distraction
                - Learn to dodge her Waterfowl Dance combo
                
                Phase 2 (Goddess of Rot):
                - She gains rot-based attacks
                - Use Cleanrot Knight summons
                - Stay away from her flower explosion
                - Same strategy but with rot damage over time
                
                FIRE GIANT:
                Phase 1:
                - Attack his left ankle (has shackle)
                - Use Torrent for mobility
                - Stay away from his fire attacks
                - Target the weak point consistently
                
                Phase 2:
                - He rolls around more - stay mobile
                - Attack his hands and eye
                - Use ranged attacks when possible
                - Alexander summon available for help
                
                GODFREY/HOARAH LOUX:
                Phase 1 (Godfrey):
                - Stay close to avoid his axe throws
                - Dodge his ground slam attacks
                - Use Spirit Summons for distraction
                - Attack during his recovery frames
                
                Phase 2 (Hoarah Loux):
                - He becomes more aggressive with grabs
                - Learn his grab patterns - they're unblockable
                - Use quickstep or bloodhound step
                - Phase has fewer openings, be patient
                
                RADAGON/ELDEN BEAST:
                Radagon:
                - Weak to physical damage
                - Dodge his hammer slams
                - Use Spirit Summons to tank
                - Learn his teleportation patterns
                
                Elden Beast:
                - Weak to physical damage
                - Stay close to avoid ranged attacks
                - Use Torrent-like mobility
                - Target the glowing spot on its side
                - Be patient - it's a long fight
                """
            },
            {
                'topic': 'build_guides',
                'source': 'curated_build_guide',
                'difficulty': 'medium',
                'type': 'character_build',
                'content': """
                Elden Ring Character Build Guide

                STRENGTH BUILDS:
                Weapon: Greatsword, Colossal Sword, Greataxe
                Stats Priority:
                - Vigor: 60 (max effective health)
                - Strength: 80 (max damage scaling)
                - Endurance: 25-30 (stamina management)
                - Mind: 15-20 (basic FP needs)
                
                Recommended Weapons:
                - Greatsword: Classic, reliable heavy weapon
                - Starscourge Greatsword: Unique skill, high damage
                - Giant-Crusher: Highest physical damage
                - Prelate's Inferno Crozier: Fire damage option
                
                Armor: Heavy armor for maximum defense
                Talismans: Great-Jar's Arsenal, Dragoncrest Shield, Green Turtle
                
                DEXTERITY BUILDS:
                Weapon: Katana, Curved Sword, Twinblade
                Stats Priority:
                - Vigor: 60
                - Dexterity: 80
                - Endurance: 30-35
                - Mind: 20-25
                
                Recommended Weapons:
                - Uchigatana: Versatile, good scaling
                - Rivers of Blood: Bleed focus, unique skill
                - Bloodhound's Fang: Curved greatsword with bleed
                - Nagakiba: Longest katana, great reach
                
                Armor: Medium armor for mobility
                Talismans: Lord of Blood's Exultation, Rotten Winged Sword, Millicent's Prosthesis
                
                INTELLIGENCE BUILDS:
                Weapon: Staff + Sorceries
                Stats Priority:
                - Vigor: 60
                - Intelligence: 80
                - Mind: 40-50
                - Endurance: 20-25
                
                Recommended Weapons:
                - Lusat's Glintstone Staff: Highest sorcery scaling
                - Meteorite Staff: Early game powerhouse
                - Carian Regal Scepter: Balanced option
                - Moonveil: INT/DEX hybrid katana
                
                Essential Sorceries:
                - Comet Azur: Massive damage beam
                - Rock Sling: High damage, tracks enemies
                - Glintstone Pebble: FP efficient basic attack
                - Terra Magica: Damage boost field
                
                FAITH BUILDS:
                Weapon: Sacred Seal + Incantations
                Stats Priority:
                - Vigor: 60
                - Faith: 80
                - Mind: 40-50
                - Endurance: 20-25
                
                Recommended Weapons:
                - Erdtree Seal: Highest faith scaling
                - Winged Scythe: Faith weapon with bleed
                - Blasphemous Blade: Life steal on kill
                - Cipher Pata: Unblockable faith scaling
                
                Essential Incantations:
                - Lightning Spear: Reliable damage
                - Rot Breath: Dragon communion spell
                - Flame of the Fell God: High damage fireball
                - Blessing's Boon: Health regeneration
                
                BLEED BUILDS:
                Weapon: Dual wielding bleed weapons
                Stats Priority:
                - Vigor: 60
                - Dexterity: 50-60
                - Arcane: 50-60
                - Endurance: 30
                
                Recommended Weapons:
                - Dual Uchigatana: Classic bleed setup
                - Dual Scavenger's Curved Sword: Fastest bleed
                - Rivers of Blood: Corpse Piler weapon skill
                - Eleonora's Poleblade: Twinblade with bleed
                
                Essential Items:
                - Seppuku Ash of War: Increases bleed buildup
                - Blood Grease: Temporary bleed boost
                - White Mask: Damage boost after bleed proc
                
                HYBRID BUILDS:
                Quality Build (STR/DEX):
                - Vigor: 60
                - Strength: 40
                - Dexterity: 40
                - Endurance: 30
                
                Spellsword (INT/DEX):
                - Vigor: 50
                - Intelligence: 60
                - Dexterity: 40
                - Mind: 30
                
                Paladin (FTH/STR):
                - Vigor: 50
                - Faith: 60
                - Strength: 40
                - Mind: 30
                
                GENERAL BUILD TIPS:
                - Always prioritize Vigor first (aim for 60)
                - Soft cap for damage stats is 80
                - Use Flask of Wondrous Physick for build synergy
                - Respec is available - experiment freely
                - Consider weapon requirements when planning
                - Plan around your preferred playstyle
                """
            },
            {
                'topic': 'exploration',
                'source': 'curated_exploration_guide',
                'difficulty': 'easy',
                'type': 'world_exploration',
                'content': """
                Elden Ring World Exploration Guide

                SITES OF GRACE:
                - Golden stakes that serve as checkpoints
                - Show golden light pointing toward main objectives
                - Rest to replenish flasks and reset enemies
                - Fast travel available between discovered sites
                - Some sites have unique NPCs or merchants
                
                TORRENT MECHANICS:
                - Double jump allows access to high areas
                - Mounted combat effective against large enemies
                - Can be summoned instantly in most areas
                - Torrent Flasks restore his health
                - Some areas require mounted exploration
                
                HIDDEN AREAS AND SECRETS:
                Illusory Walls:
                - Hit walls that look suspicious or different
                - Often found near messages saying "hidden path ahead"
                - Some require multiple hits to reveal
                - Check behind altars and in dead-end corridors
                
                Catacombs and Dungeons:
                - Small dungeon areas with bosses
                - Usually contain valuable loot and upgrade materials
                - Lever puzzles common in catacombs
                - Boss fights often have unique rewards
                
                WORLD AREAS:
                Limgrave (Starting Area):
                - Relatively safe, good for learning mechanics
                - Gatefront Ruins: Early game hub
                - Mistwood: Blaidd questline starts here
                - Caelid border: High-level area, avoid early
                
                Liurnia of the Lakes:
                - Water-filled area with Academy of Raya Lucaria
                - Carian Manor: Ranni questline location
                - Many sorcery-related items and NPCs
                - Rennala boss fight in the Academy
                
                Caelid:
                - High-level area with scarlet rot
                - Redmane Castle: Radahn boss fight
                - Dragonbarrow: Even higher level sub-area
                - Excellent for high-level farming
                
                Altus Plateau:
                - Mid-game area, accessed via lift or ravine
                - Leyndell Royal Capital: Major story location
                - Volcano Manor: Rykard boss and questlines
                - Mt. Gelmir: Challenging mountainous area
                
                Mountaintops of the Giants:
                - Late game area, very challenging
                - Fire Giant boss fight
                - Leads to Crumbling Farum Azula
                - Haligtree access point
                
                COLLECTIBLES AND ITEMS:
                Golden Seeds:
                - Upgrade Sacred Flask potency
                - Found near golden saplings
                - Also dropped by some bosses
                - Essential for survivability
                
                Sacred Tears:
                - Upgrade Sacred Flask uses
                - Found in churches and ruins
                - Guarded by tough enemies
                - Priority items for exploration
                
                Memory Stones:
                - Increase memory slots for spells
                - Found in towers and specific locations
                - Essential for magic builds
                - Some require puzzle solving
                
                EXPLORATION TIPS:
                - Use telescope to scout areas safely
                - Follow roads to find points of interest
                - Investigate ruins and structures
                - Talk to NPCs multiple times
                - Return to areas after story progression
                - Use map markers to track discoveries
                - Stock up on arrows for switches
                - Some areas change based on story progress
                
                NAVIGATION TIPS:
                - Golden grace light points to main objectives
                - Use map markers for personal waypoints
                - Stake of Marika appear near boss fights
                - Some areas require specific story progression
                - Weather can affect visibility and enemy behavior
                - Time of day affects some NPC availability
                """
            }
        ]
    
    def get_witcher3_guides(self) -> List[Dict]:
        """The Witcher 3 comprehensive guides"""
        return [
            {
                'topic': 'combat_system',
                'source': 'curated_combat_guide',
                'difficulty': 'medium',
                'type': 'combat',
                'content': """
                The Witcher 3 Combat System Guide

                SWORD COMBAT:
                Steel Sword: Use against humans and non-monsters
                Silver Sword: Use against monsters and supernatural enemies
                Fast Attacks: Quick strikes, good for combos
                Strong Attacks: Slower but more damage, breaks guards
                
                SIGNS (Magic):
                Igni: Fire damage, effective against multiple enemies
                - Upgrades: Intense heat, firestream
                - Best against: Necrophages, vampires, wraiths
                
                Quen: Protective shield, absorbs damage
                - Upgrades: Exploding shield, healing
                - Essential for survival on higher difficulties
                
                Aard: Telekinetic blast, knocks down enemies
                - Upgrades: Sweeping blast, piercing
                - Great for crowd control and environmental kills
                
                Yrden: Magic trap, slows enemies
                - Upgrades: Multiple traps, damage over time
                - Effective against fast enemies and wraiths
                
                Axii: Mind control, stuns enemies
                - Upgrades: Puppet, group control
                - Useful in conversations and against humans
                
                COMBAT STRATEGIES:
                Dodge vs. Roll:
                - Dodge: Quick sidestep, keeps you close
                - Roll: Longer distance, uses more stamina
                - Use dodge for single enemies, roll for groups
                
                Parrying and Counterattacks:
                - Time blocks perfectly for counterattacks
                - Works best against human enemies
                - Less effective against monsters
                
                Preparation:
                - Apply blade oils for damage bonuses
                - Use potions before tough fights
                - Repair weapons and armor regularly
                - Stock up on food for healing
                
                ENEMY TYPES:
                Humans: Use steel sword, parry and counter
                Nekkers: Group enemies, use Igni and Aard
                Wraiths: Use Yrden to slow, then silver sword
                Vampires: Use Igni and vampire oil
                Golems: Attack from behind, use Quen
                Dragons: Stay mobile, use crossbow for flying attacks
                
                ADVANCED TECHNIQUES:
                Whirl: Spinning attack, great for groups
                Rend: Charged attack, massive damage
                Crossbow: Ranged option, essential for flying enemies
                Bombs: Crowd control and elemental damage
                """
            },
            {
                'topic': 'quest_walkthrough',
                'source': 'curated_quest_guide',
                'difficulty': 'medium',
                'type': 'story_quests',
                'content': """
                The Witcher 3 Major Quest Guide

                MAIN STORY QUESTS:
                
                Bloody Baron Questline:
                - Find Anna and Tamara in Velen
                - Family Matters quest has multiple outcomes
                - Choices affect Baron's fate and his family
                - Recommended: Save the spirit in the tree for best outcome
                
                Novigrad Questlines:
                - Find Dandelion through various contacts
                - Triss romance option available
                - Choices affect mage persecution storyline
                - Help or hinder witch hunters based on preference
                
                Skellige Main Quest:
                - Find Ciri through Uma transformation
                - Complete trials at Kaer Morhen
                - Gather allies for final battle
                - Choices here affect Ciri's fate
                
                IMPORTANT SIDE QUESTS:
                
                Witcher Contracts:
                - High-paying monster hunts
                - Require investigation and preparation
                - Use Witcher senses to gather clues
                - Apply appropriate oils and potions
                
                Gwent Quests:
                - Collect Gwent cards from merchants
                - Win unique cards from specific NPCs
                - Complete Gwent tournaments
                - Build powerful decks for tournaments
                
                Romance Options:
                Triss Merigold:
                - Complete "A Matter of Life and Death"
                - Choose romantic dialogue options
                - Help her in "Now or Never" quest
                
                Yennefer of Vengerberg:
                - Complete "The Last Wish" quest
                - Choose romantic dialogue options
                - Help with various magical tasks
                
                QUEST TIPS:
                - Read quest descriptions carefully
                - Use Witcher senses frequently
                - Talk to all NPCs for additional information
                - Save before making major decisions
                - Some quests have time limits
                - Explore thoroughly for hidden objectives
                """
            },
            {
                'topic': 'character_builds',
                'source': 'curated_build_guide',
                'difficulty': 'medium',
                'type': 'character_progression',
                'content': """
                The Witcher 3 Character Build Guide

                COMBAT BUILDS:
                
                Fast Attack Build:
                Skills to Focus:
                - Muscle Memory: Increases fast attack damage
                - Precise Blows: Increases critical hit chance
                - Whirl: Spinning attack for groups
                - Crippling Strikes: Chance to cripple enemies
                
                Recommended Equipment:
                - Cat School Gear: Bonuses to fast attacks
                - Severance Runeword: Increases Whirl range
                - Preservation Runeword: Reduces armor degradation
                
                Strong Attack Build:
                Skills to Focus:
                - Strength Training: Increases strong attack damage
                - Crushing Blows: Increases critical hit damage
                - Rend: Charged attack ability
                - Sunder Armor: Reduces enemy armor
                
                Recommended Equipment:
                - Griffin School Gear: Balanced stats
                - Bear School Gear: Heavy armor protection
                - Severance Runeword: Increases Rend range
                
                SIGN BUILDS:
                
                Igni Build:
                Skills to Focus:
                - Melt Armor: Igni reduces enemy armor
                - Firestream: Alternative Igni mode
                - Pyromaniac: Increases burning damage
                - Intense Heat: Increases Igni damage
                
                Recommended Equipment:
                - Griffin School Gear: Sign intensity bonus
                - Igni Runestones: Increases fire damage
                - Superior Tawny Owl: Reduces Sign stamina cost
                
                Quen Build:
                Skills to Focus:
                - Exploding Shield: Quen damages nearby enemies
                - Active Shield: Alternative Quen mode
                - Quen Intensity: Increases shield strength
                - Magic Trap: Yrden affects more enemies
                
                ALCHEMY BUILDS:
                
                Toxicity Build:
                Skills to Focus:
                - Acquired Tolerance: Increases toxicity threshold
                - Tissue Transmutation: Toxicity increases damage
                - Synergy: Mutagen effects are enhanced
                - Killing Spree: Kills reduce toxicity
                
                Recommended Equipment:
                - Manticore School Gear: Alchemy bonuses
                - Superior Swallow: Enhanced healing
                - Superior Thunderbolt: Increased damage
                
                HYBRID BUILDS:
                
                Balanced Build:
                - Distribute points across combat and signs
                - Focus on survivability skills
                - Use situational signs and sword techniques
                - Recommended for first playthrough
                
                Mutation Builds (Blood and Wine):
                - Euphoria: Toxicity increases damage
                - Piercing Cold: Aard can instantly kill
                - Conductors of Magic: Signs can critical hit
                - Metamorphosis: Mutagens grant new abilities
                
                GENERAL TIPS:
                - Use Place of Power to gain extra skill points
                - Respec is available with Potion of Clearance
                - Equipment sets provide significant bonuses
                - Mutagens enhance skill effects
                - Consider your playstyle when choosing builds
                - Higher difficulties favor defensive builds
                """
            }
        ]
    
    # Add more games here...
    def get_dark_souls_guides(self) -> List[Dict]:
        """Dark Souls 3 guides"""
        return [
            {
                'topic': 'boss_strategies',
                'source': 'curated_boss_guide',
                'difficulty': 'hard',
                'type': 'combat',
                'content': """
                Dark Souls 3 Boss Strategy Guide

                IUDEX GUNDYR (Tutorial Boss):
                - Straightforward fight to learn basics
                - Phase 2: Black serpent emerges, stay behind him
                - Learn to dodge roll and time attacks
                - Use firebombs for extra damage
                
                VORDT OF THE BOREAL VALLEY:
                - Stay behind him and attack his back legs
                - Phase 2: He becomes more aggressive with charge attacks
                - Use the pillars as cover
                - Frostbite resistance helpful but not required
                
                ABYSS WATCHERS:
                Phase 1: Fight multiple Watchers
                - Let them fight each other
                - Focus on one when others are distracted
                - Red-eyed Watcher helps you
                
                Phase 2: Single Watcher with fire
                - More aggressive with wider attacks
                - Stay close to avoid fire trail attacks
                - Roll through his combo attacks
                
                PONTIFF SULYVAHN:
                - Extremely aggressive boss
                - Phase 1: Learn his combo patterns
                - Phase 2: He summons a clone
                - Use fast weapons for quick hits
                - Parrying is possible but risky
                
                ALDRICH DEVOURER OF GODS:
                - Teleports around the arena
                - Arrow rain: Run perpendicular to avoid
                - Soul spear: Dodge roll at the right moment
                - Attack his tail/lower body for damage
                - Fire resistance recommended
                
                DANCER OF THE BOREAL VALLEY:
                - Weak to Dark damage
                - Phase 1: Relatively slow, learn patterns
                - Phase 2: Dual swords, much faster
                - Stay close to her back
                - Long reach attacks, stay mobile
                
                DRAGONSLAYER ARMOR:
                - Lightning and Physical damage
                - Use the environment: Pilgrim Butterflies
                - Stay behind him when possible
                - Phase 2: More aggressive with magic support
                - Lightning resistance helpful
                
                TWIN PRINCES:
                Phase 1: Lothric Prince
                - Teleports frequently
                - Magic attacks from range
                - Close distance quickly
                
                Phase 2: Lorian carries Lothric
                - Combined attacks
                - Lorian's sword attacks + Lothric's magic
                - Kill Lorian first, then Lothric quickly
                
                SOUL OF CINDER (Final Boss):
                - Multiple phases with different movesets
                - Phase 1: Cycles through weapon types
                - Phase 2: Gwyn's moveset with new attacks
                - Learn each weapon style
                - Parrying possible in phase 2
                
                NAMELESS KING (Optional):
                Phase 1: King of Storms (Dragon)
                - Target the dragon's head
                - Use ranged attacks or magic
                - Stay locked onto the head
                
                Phase 2: Nameless King
                - Extremely difficult
                - Lightning attacks with delayed timing
                - Use lightning resistance
                - Stay close to avoid lightning spears
                """
            }
        ]
    
    # Simplified versions for remaining games
    def get_sekiro_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'hard', 'type': 'combat', 'content': 'Sekiro combat focuses on parrying and posture breaking. Learn to deflect attacks perfectly and use prosthetic tools strategically.'}]
    
    def get_cyberpunk_guides(self) -> List[Dict]:
        return [{'topic': 'builds', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'character_build', 'content': 'Cyberpunk 2077 builds focus on Netrunner (hacking), Solo (combat), or Techie (crafting) playstyles. Allocate attribute points accordingly.'}]
    
    def get_ac_valhalla_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'combat', 'content': 'AC Valhalla combat uses heavy and light attacks. Master dodging, parrying, and use abilities strategically in raids.'}]
    
    def get_cod_warzone_guides(self) -> List[Dict]:
        return [{'topic': 'strategy', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'multiplayer', 'content': 'Warzone strategy: Choose good landing spots, manage the gas zone, use buy stations strategically, and communicate with your team.'}]
    
    def get_destiny2_guides(self) -> List[Dict]:
        return [{'topic': 'builds', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'character_build', 'content': 'Destiny 2 builds focus on subclass synergy, weapon combinations, and armor stats. Match your playstyle with appropriate mods and exotic gear.'}]
    
    def get_horizon_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'combat', 'content': 'Horizon combat: Use different arrow types for different machines, target weak points, and use traps and environmental hazards.'}]
    
    def get_ghost_tsushima_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'combat', 'content': 'Ghost of Tsushima: Master different stances for different enemy types, use stealth and ghost weapons, perfect your parrying timing.'}]
    
    def get_bloodborne_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'hard', 'type': 'combat', 'content': 'Bloodborne: Aggressive combat with rally system. Use trick weapons transformations, manage blood vials, and learn boss patterns.'}]
    
    def get_nioh2_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'hard', 'type': 'combat', 'content': 'Nioh 2: Manage ki (stamina), use different weapon stances, master burst counters, and utilize yokai abilities strategically.'}]
    
    def get_mhw_guides(self) -> List[Dict]:
        return [{'topic': 'combat', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'combat', 'content': 'Monster Hunter World: Learn monster patterns, use appropriate weapons, manage stamina, and coordinate with team members.'}]
    
    def get_rdr2_guides(self) -> List[Dict]:
        return [{'topic': 'gameplay', 'source': 'curated_guide', 'difficulty': 'easy', 'type': 'general', 'content': 'Red Dead Redemption 2: Manage honor system, take care of your horse, hunt animals for upgrades, and explore the open world thoroughly.'}]
    
    def get_gtav_guides(self) -> List[Dict]:
        return [{'topic': 'gameplay', 'source': 'curated_guide', 'difficulty': 'easy', 'type': 'general', 'content': 'GTA V: Switch between three characters, complete heists strategically, invest in the stock market, and explore Los Santos.'}]
    
    def get_minecraft_guides(self) -> List[Dict]:
        return [{'topic': 'crafting', 'source': 'curated_guide', 'difficulty': 'easy', 'type': 'crafting', 'content': 'Minecraft: Learn crafting recipes, mine resources efficiently, build shelters, and explore different biomes for materials.'}]
    
    def get_fortnite_guides(self) -> List[Dict]:
        return [{'topic': 'strategy', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'battle_royale', 'content': 'Fortnite: Master building mechanics, choose good landing spots, manage resources, and adapt to the shrinking storm circle.'}]
    
    def get_apex_guides(self) -> List[Dict]:
        return [{'topic': 'strategy', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'battle_royale', 'content': 'Apex Legends: Choose legends based on team composition, master movement mechanics, use the ping system, and control positioning.'}]
    
    def get_valorant_guides(self) -> List[Dict]:
        return [{'topic': 'strategy', 'source': 'curated_guide', 'difficulty': 'medium', 'type': 'tactical_fps', 'content': 'Valorant: Learn agent abilities, master economy management, practice crosshair placement, and communicate with your team effectively.'}]
    
    def query_game(self, game_name: str, question: str):
        """Query a specific game's knowledge base"""
        return self.rag_engine.query(question, game_filter=game_name)
    
    def list_loaded_games(self):
        """List all loaded games"""
        return self.games_loaded
    
    def get_game_stats(self):
        """Get statistics about loaded games"""
        return self.rag_engine.get_stats()

def main():
    """Load all games and test the system"""
    print("Loading Pre-Curated Game Database...")
    
    # Initialize and load all games
    preloaded = PreloadedGames()
    loaded_games = preloaded.load_all_games()
    
    # Show system stats
    stats = preloaded.get_game_stats()
    print(f"\nSystem Ready!")
    print(f"Games loaded: {len(loaded_games)}")
    print(f"Total documents: {stats['vector_store']['document_count']}")
    print(f"Available for query: {stats['available_games']}")
    
    # Test queries
    print("\n" + "="*50)
    print("TESTING QUERIES")
    print("="*50)
    
    test_queries = [
        ("God of War", "How do I defeat Baldur?"),
        ("Elden Ring", "What's the best build for beginners?"),
        ("The Witcher 3", "How do I use Quen effectively?"),
        ("Dark Souls 3", "Tips for fighting Pontiff Sulyvahn?"),
    ]
    
    for game, question in test_queries:
        print(f"\n{game}: {question}")
        response = preloaded.query_game(game, question)
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
    
    print("\n" + "="*50)
    print("PRE-LOADED GAME DATABASE READY!")
    print("="*50)

if __name__ == "__main__":
    main()