import re
from typing import Dict, List, Tuple


class FootballTacticsPreprocessor:
    """Preprocesses text to normalize football terminology and extract domain entities."""
    
    KEY_FIGURES = [
        'arsene wenger', 'alex ferguson', 'eric cantona', 'dennis bergkamp',
        'thierry henry', 'alan ball', 'matt le tissier', 'gary neville',
        'ian wright', 'patrick vieira', 'roy hodgson', 'harry redknapp',
        'brian clough', 'don revie', 'george graham', 'bruce rioch',
        'kenny dalglish', 'bob paisley', 'bill shankly', 'ruud van nistelrooy',
        'wayne rooney', 'cristiano ronaldo', 'rio ferdinand', 'ashley cole',
        'sol campbell', 'kolo toure', 'gilberto silva', 'robert pires',
        'freddie ljungberg', 'david platt', 'peter schmeichel',
        'gareth bale', 'peter crouch', 'didier drogba', 'frank lampard'
        'roman abramovich', 'arsene wenger', 'david dein', 'peter hill-wood',
        'glazer family', 'murdoch', 'silver lake', 'kroenke',
        'mike ashley', 'daniel levy', 'jack walker', 'sheikh mansour',
        'rupert murdoch', 'bskyb', 'virgin', 'emirates airline'
    ]
    
    # Key teams
    KEY_TEAMS = [
        'arsenal', 'manchester united', 'liverpool', 'chelsea', 'manchester city',
        'tottenham', 'newcastle', 'southampton', 'aston villa', 'everton',
        'bolton', 'stoke', 'portsmouth', 'fulham', 'qpr', 'leeds',
        'leicester', 'burnley', 'blackburn', 'wimbledon', 'sunderland'
    ]
    
    # Formation patterns
    FORMATION_PATTERNS = {
        r'4\s*[-–]\s*4\s*[-–]\s*2': '4-4-2',
        r'442': '4-4-2',
        r'4/4/2': '4-4-2',
        r'4\s*[-–]\s*3\s*[-–]\s*3': '4-3-3',
        r'433': '4-3-3',
        r'4/3/3': '4-3-3',
        r'4\s*[-–]\s*2\s*[-–]\s*3\s*[-–]\s*1': '4-2-3-1',
        r'4231': '4-2-3-1',
        r'4/2/3/1': '4-2-3-1',
        r'3\s*[-–]\s*5\s*[-–]\s*2': '3-5-2',
        r'352': '3-5-2',
        r'3/5/2': '3-5-2',
        r'5\s*[-–]\s*3\s*[-–]\s*2': '5-3-2',
        r'532': '5-3-2',
        r'5\s*[-–]\s*2\s*[-–]\s*3': '5-2-3',
        r'523': '5-2-3',
        r'W[-–]M': 'W-M formation',
        r'back\s+four': 'back-four system',
        r'back\s+three': 'back-three system',
    }
    
    # Tactical concepts
    TACTICAL_SYNONYMS = {
        'cdm': 'defensive midfielder',
        'cam': 'attacking midfielder',
        'cm': 'central midfielder',
        'lwb': 'left wing-back',
        'rwb': 'right wing-back',
        'lb': 'left-back',
        'rb': 'right-back',
        'cb': 'center-back',
        'cf': 'center-forward',
        'st': 'striker',
        'lw': 'left winger',
        'rw': 'right winger',
        'gegenpressing': 'high pressing system',
        'gegenpress': 'high pressing system',
        'counter-press': 'high pressing system',
        'false 9': 'false number 9 playmaker',
        'deep-lying playmaker': 'regista role',
        'libero': 'sweeper libero',
        'regista': 'deep playmaker midfielder',
        'trequartista': 'attacking midfielder behind striker',
        'enganche': 'attacking midfielder playmaker',
        'volante': 'box-to-box midfielder',
        'mezzala': 'interior midfielder',
        'tiki-taka': 'possession-based passing system',
        'catenaccio': 'defensive zone system',
        'total football': 'positional interchangeable system',
        'pressing trap': 'coordinated pressing trigger',
    }
    
    DEFENSIVE_CONCEPTS = {
        'width': 'lateral defensive coverage',
        'compactness': 'defensive density and spacing',
        'pressing trigger': 'moment to initiate press',
        'defensive line': 'defensive line depth and positioning',
        'offside trap': 'coordinated offside line raising',
        'defensive shape': 'defensive formation positioning',
        'cover shadow': 'positioning behind opponent to block pass',
        'pressing angle': 'approach angle for pressing defender',
    }
    
    OFFENSIVE_CONCEPTS = {
        'buildup play': 'possession progression from defense',
        'transition play': 'quick counter-attack after possession loss',
        'third man run': 'movement off ball for passing option',
        'false movement': 'dummy run to create space',
        'inside run': 'run cutting inside from wing position',
        'width': 'attacking width through wingers/fullbacks',
        'penetration': 'vertical passing forward through lines',
        'overload': 'numerical advantage in specific area',
    }
    
    def __init__(self):
        """Initialize preprocessor with compiled patterns."""
        self.formation_regex = {
            pattern: re.compile(pattern, re.IGNORECASE)
            for pattern in self.FORMATION_PATTERNS.keys()
        }
    
    @staticmethod
    def normalize_formations(text: str) -> str:
        """Normalize all formation notation variations to standard format."""
        processed = text
        for pattern, standard_form in FootballTacticsPreprocessor.FORMATION_PATTERNS.items():
            regex = re.compile(pattern, re.IGNORECASE)
            processed = regex.sub(standard_form, processed)
        return processed
    
    @staticmethod
    def normalize_tactical_terms(text: str) -> str:
        """Normalize tactical terminology while preserving context."""
        processed = text
        sorted_synonyms = sorted(
            FootballTacticsPreprocessor.TACTICAL_SYNONYMS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        for abbrev, full_term in sorted_synonyms:
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            processed = re.sub(pattern, full_term, processed, flags=re.IGNORECASE)
        return processed
    
    @staticmethod
    def extract_tactical_entities(text: str) -> Dict[str, List[str]]:
        """Extract tactical entities by category."""
        entities = {
            'formations': [],
            'defensive_concepts': [],
            'offensive_concepts': [],
            'tactical_roles': [],
            'key_figures': [],  
            'key_teams': []  
        }
        
        text_lower = text.lower()
        
        for formation in ['4-4-2', '4-3-3', '4-2-3-1', '3-5-2', '5-2-3', '5-3-2', 'W-M', '2-3-5']:
            if formation.lower() in text_lower:
                entities['formations'].append(formation)
        
        for concept in FootballTacticsPreprocessor.DEFENSIVE_CONCEPTS.keys():
            pattern = r'\b' + re.escape(concept.lower()) + r'\b'
            if re.search(pattern, text_lower):
                entities['defensive_concepts'].append(concept)
        
        for concept in FootballTacticsPreprocessor.OFFENSIVE_CONCEPTS.keys():
            pattern = r'\b' + re.escape(concept.lower()) + r'\b'
            if re.search(pattern, text_lower):
                entities['offensive_concepts'].append(concept)
        
        roles = ['regista', 'trequartista', 'libero', 'enganche', 'mezzala', 'volante',
                 'defensive midfielder', 'attacking midfielder', 'false number 9']
        for role in roles:
            pattern = r'\b' + re.escape(role.lower()) + r'\b'
            if re.search(pattern, text_lower):
                entities['tactical_roles'].append(role)
        
        for figure in FootballTacticsPreprocessor.KEY_FIGURES:
            pattern = r'\b' + re.escape(figure.lower()) + r'\b'
            if re.search(pattern, text_lower):
                entities['key_figures'].append(figure.title())
        
        for team in FootballTacticsPreprocessor.KEY_TEAMS:
            pattern = r'\b' + re.escape(team.lower()) + r'\b'
            if re.search(pattern, text_lower):
                entities['key_teams'].append(team.title())
        
        return entities
    
    def preprocess_chunk(self, content: str) -> Tuple[str, Dict]:
        """Complete preprocessing of a text chunk for football domain."""
        processed = self.normalize_formations(content)
        processed = self.normalize_tactical_terms(processed)
        entities = self.extract_tactical_entities(processed)
        return processed, entities
    
    @staticmethod
    def create_tactical_keywords(entities: Dict[str, List[str]]) -> str:
        """Create enhanced keyword string for better embedding."""
        keywords = []
        
        if entities.get('key_figures'):
            keywords.append(f"figures: {', '.join(entities['key_figures'])}")
        
        if entities.get('key_teams'):
            keywords.append(f"teams: {', '.join(entities['key_teams'])}")
        
        if entities.get('formations'):
            keywords.append(f"formations: {', '.join(entities['formations'])}")
        
        if entities.get('tactical_roles'):
            keywords.append(f"tactical roles: {', '.join(entities['tactical_roles'])}")
        
        if entities.get('defensive_concepts'):
            keywords.append(f"defensive tactics: {', '.join(entities['defensive_concepts'])}")
        
        if entities.get('offensive_concepts'):
            keywords.append(f"offensive tactics: {', '.join(entities['offensive_concepts'])}")
        
        return " | ".join(keywords)
