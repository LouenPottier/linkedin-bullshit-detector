// ============================================================
// LinkedIn Bullshit Detector — rules.js
// ============================================================
// Ce fichier contient toutes les règles de détection.
// Tu peux le modifier librement sans toucher au reste du code.
// ============================================================

const BSD_RULES = {

  // ----------------------------------------------------------
  // RÈGLE 1 : Mots-clés dans le texte du post
  // ----------------------------------------------------------
  keywords: [
    // FR - empowerment / dev perso
    "bienveillance", "résilience", "mindset", "leadership", "synergie", "synergies",
    "agilité", "disruption", "disruptif", "disruptive", "game changer", "game-changer",
    "passion", "passionné", "passionnée", "inspirant", "inspirante", "humilité",
    "authenticité", "vulnérabilité", "audace", "impact", "impactant", "valeurs",
    "mission", "vision", "purpose", "épanouissement", "zone de confort", "ikigai",
    "storytelling", "personal branding", "soft skills", "hard skills",
    // FR - corporate
    "holistique", "transversal", "pivoter", "scalable", "scalabilité", "onboarding",
    "quick win", "pain point", "best practice", "benchmark", "roadmap", "framework",
    "optimisation", "excellence", "transformation digitale", "innovation", "innovant",
    "écosystème", "performer", "digitalisation",
    // FR - patterns narratifs
    "j'ai réalisé", "ce que personne ne vous dit", "la vérité sur", "voici pourquoi",
    "le secret", "j'aurais aimé savoir", "leçon de vie", "partager mon expérience",
    "je suis fier", "je suis fière", "je suis ravi", "je suis ravie",
    "je suis honoré", "opportunité", "chance incroyable", "humble",
    // EN - empowerment
    "resilience", "hustle", "grind", "entrepreneur", "entrepreneurship",
    "inspiring", "inspirational", "authenticity", "vulnerability", "boldness",
    "impactful", "comfort zone", "thought leader", "thought leadership", "growth mindset",
    // EN - corporate
    "leverage", "leveraging", "pivot", "scalable", "scalability",
    "best practices", "kpi", "okr", "agile", "lean", "scrum", "sprint",
    "deep dive", "bandwidth", "move the needle", "low-hanging fruit",
    "circle back", "ecosystem", "digital transformation",
    // EN - patterns narratifs
    "excited to announce", "happy to share", "happy to announce",
    "pleased to announce", "incredibly honored", "thrilled", "blessed",
    "grateful", "gratitude", "life lesson", "what nobody tells you",
    "the truth about", "here's why", "i wish i knew",
  ],

  // ----------------------------------------------------------
  // RÈGLE 2 : Intitulé de l'auteur (headline)
  // Approche structurelle : on détecte les patterns, pas des mots spécifiques.
  // Un intitulé factuel = titre + entreprise (court, sans pronom).
  // Un intitulé bullshit = phrase commerciale avec pronoms, longue, avec liste.
  // ----------------------------------------------------------
  headline: {

    // Mots-clés spécifiques toujours suspects dans un intitulé
    bullshitWords: [
      "™", "®",
      "solopreneur", "intrapreneur", "slasheur",
      "ninja", "guru", "wizard", "rockstar", "évangéliste", "evangelist",
      "thought leader", "visionnaire", "serial entrepreneur",
      "game changer", "disrupteur",
    ],

    // Poids des différentes règles (ajuste selon tes observations)
    pronounPenalty: 4,   // Pronom personnel dans l'intitulé
    keywordPenalty: 2,   // Mot-clé bullshit dans l'intitulé
    lengthPenalty: 1,    // Intitulé > 80 caractères
    colonPenalty: 1,     // ":" suivi d'une liste de services
  },

};
