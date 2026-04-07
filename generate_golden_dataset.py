"""
Synthetic Golden Dataset Generator for Financial Reasoning Evaluation
======================================================================

The dataset includes:
- 5 Macro events (monetary policy, economic indicators)
- 5 Earnings Misses (company performance disappointments)
- 5 Geopolitical Shocks (international conflicts, policy changes)
- 5 Noise/Irrelevant events (should result in Neutral predictions)

"""

import json
from typing import List, Dict, Literal
from datetime import datetime


def generate_golden_dataset() -> List[Dict]:
    """
    Generate 20 synthetic market event examples for evaluation.
    
    Returns:
        List of dictionaries containing market event data
    """
    
    dataset = []
    
    macro_events = [
        {
            "id": "001",
            "event_category": "Macro",
            "input_news": "Fed Chair Powell signals 50bps rate hike due to persistent inflation.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Cost of capital increases",
                "Discount rate rises",
                "Growth stocks hit hardest"
            ],
            "market_regime": "Hawkish_Policy"
        },
        {
            "id": "002",
            "event_category": "Macro",
            "input_news": "US GDP growth accelerates to 3.2% QoQ, beating expectations of 2.5%.",
            "input_volatility": "Low",
            "expected_direction": "Bullish",
            "key_reasoning_points": [
                "Strong economic expansion",
                "Corporate earnings likely to improve",
                "Consumer spending robust"
            ],
            "market_regime": "Economic_Expansion"
        },
        {
            "id": "003",
            "event_category": "Macro",
            "input_news": "ECB announces emergency bond-buying program amid sovereign debt crisis.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Systemic risk in European banking",
                "Currency devaluation pressure",
                "Flight to quality assets expected"
            ],
            "market_regime": "Crisis_Mode"
        },
        {
            "id": "004",
            "event_category": "Macro",
            "input_news": "Unemployment rate drops to 3.5%, lowest in 50 years, wage growth accelerates.",
            "input_volatility": "Medium",
            "expected_direction": "Mixed",
            "key_reasoning_points": [
                "Tight labor market positive for consumer spending",
                "Wage inflation may pressure corporate margins",
                "Fed may tighten policy further"
            ],
            "market_regime": "Late_Cycle"
        },
        {
            "id": "005",
            "event_category": "Macro",
            "input_news": "China announces $500B stimulus package to boost domestic consumption.",
            "input_volatility": "Medium",
            "expected_direction": "Bullish",
            "key_reasoning_points": [
                "Global demand for commodities increases",
                "Multinational corporations benefit from China exposure",
                "Risk-on sentiment improves"
            ],
            "market_regime": "Stimulus_Driven"
        }
    ]
    
    # ========================================================================
    # EARNINGS MISSES (5 examples)
    # ========================================================================
    
    earnings_events = [
        {
            "id": "006",
            "event_category": "Earnings_Miss",
            "input_news": "Tesla Q3 earnings miss by 15%, deliveries fall short due to production issues.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Revenue guidance lowered",
                "Operational execution concerns",
                "Competitive pressure in EV market"
            ],
            "market_regime": "Sector_Weakness"
        },
        {
            "id": "007",
            "event_category": "Earnings_Miss",
            "input_news": "Meta reports 20% decline in ad revenue, misses EPS by $0.45.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Digital advertising slowdown",
                "User engagement metrics declining",
                "Metaverse investments not paying off"
            ],
            "market_regime": "Tech_Correction"
        },
        {
            "id": "008",
            "event_category": "Earnings_Miss",
            "input_news": "Boeing misses earnings estimates, 787 production delays continue.",
            "input_volatility": "Medium",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Supply chain disruptions persist",
                "Cash burn rate elevated",
                "Regulatory scrutiny intensifies"
            ],
            "market_regime": "Industrial_Slowdown"
        },
        {
            "id": "009",
            "event_category": "Earnings_Miss",
            "input_news": "Walmart Q4 earnings miss, same-store sales decline 2.3% amid inflation.",
            "input_volatility": "Medium",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Consumer spending weakening",
                "Margin compression from discounting",
                "Inventory buildup signals demand slowdown"
            ],
            "market_regime": "Consumer_Weakness"
        },
        {
            "id": "010",
            "event_category": "Earnings_Miss",
            "input_news": "Netflix loses 1M subscribers, misses revenue forecast by 8%.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Subscriber churn accelerating",
                "Streaming competition intensifies",
                "Content costs remain elevated"
            ],
            "market_regime": "Growth_Deceleration"
        }
    ]
    
    # ========================================================================
    # GEOPOLITICAL SHOCKS (5 examples)
    # ========================================================================
    
    geopolitical_events = [
        {
            "id": "011",
            "event_category": "Geopolitical_Shock",
            "input_news": "Russia invades Ukraine, oil prices spike 25% to $120/barrel.",
            "input_volatility": "Extreme",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Energy supply disruption",
                "Sanctions impact global trade",
                "Safe-haven flows to bonds and gold"
            ],
            "market_regime": "Geopolitical_Crisis"
        },
        {
            "id": "012",
            "event_category": "Geopolitical_Shock",
            "input_news": "US-China trade war escalates, 25% tariffs imposed on $200B of goods.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Supply chain disruptions",
                "Corporate profit margins compressed",
                "Global growth outlook deteriorates"
            ],
            "market_regime": "Trade_War"
        },
        {
            "id": "013",
            "event_category": "Geopolitical_Shock",
            "input_news": "Middle East conflict intensifies, Strait of Hormuz shipping threatened.",
            "input_volatility": "Extreme",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Oil supply route at risk",
                "Energy prices surge",
                "Global recession fears mount"
            ],
            "market_regime": "Energy_Crisis"
        },
        {
            "id": "014",
            "event_category": "Geopolitical_Shock",
            "input_news": "North Korea conducts nuclear test, UN Security Council convenes emergency session.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "Regional instability increases",
                "Defense stocks may benefit",
                "Risk-off sentiment dominates"
            ],
            "market_regime": "Security_Threat"
        },
        {
            "id": "015",
            "event_category": "Geopolitical_Shock",
            "input_news": "Brexit negotiations collapse, UK faces hard exit from EU.",
            "input_volatility": "High",
            "expected_direction": "Bearish",
            "key_reasoning_points": [
                "GBP currency crisis",
                "European financial sector exposed",
                "Trade uncertainty impacts multinationals"
            ],
            "market_regime": "Political_Uncertainty"
        }
    ]
    
    # ========================================================================
    # NOISE/IRRELEVANT EVENTS (5 examples)
    # ========================================================================
    
    noise_events = [
        {
            "id": "016",
            "event_category": "Noise",
            "input_news": "Apple CEO Tim Cook spotted at new coffee shop in Cupertino.",
            "input_volatility": "Low",
            "expected_direction": "Neutral",
            "key_reasoning_points": [
                "No material business impact",
                "Celebrity sighting irrelevant to fundamentals",
                "Market should ignore this noise"
            ],
            "market_regime": "Normal_Operations"
        },
        {
            "id": "017",
            "event_category": "Noise",
            "input_news": "Tesla changes office furniture supplier, no financial impact disclosed.",
            "input_volatility": "Low",
            "expected_direction": "Neutral",
            "key_reasoning_points": [
                "Immaterial operational change",
                "No revenue or cost implications",
                "Not relevant for investment decisions"
            ],
            "market_regime": "Normal_Operations"
        },
        {
            "id": "018",
            "event_category": "Noise",
            "input_news": "Amazon warehouse workers vote on break room vending machine options.",
            "input_volatility": "Low",
            "expected_direction": "Neutral",
            "key_reasoning_points": [
                "Trivial internal matter",
                "Zero impact on business model",
                "Media noise without substance"
            ],
            "market_regime": "Normal_Operations"
        },
        {
            "id": "019",
            "event_category": "Noise",
            "input_news": "Microsoft updates company logo color from #0078D4 to #0078D7.",
            "input_volatility": "Low",
            "expected_direction": "Neutral",
            "key_reasoning_points": [
                "Cosmetic branding change",
                "No financial materiality",
                "Irrelevant to stock valuation"
            ],
            "market_regime": "Normal_Operations"
        },
        {
            "id": "020",
            "event_category": "Noise",
            "input_news": "Google cafeteria introduces new vegan menu options for employees.",
            "input_volatility": "Low",
            "expected_direction": "Neutral",
            "key_reasoning_points": [
                "Employee perk with negligible cost",
                "No impact on competitive position",
                "Not material to investment thesis"
            ],
            "market_regime": "Normal_Operations"
        }
    ]
    
    # Combine all events
    dataset.extend(macro_events)
    dataset.extend(earnings_events)
    dataset.extend(geopolitical_events)
    dataset.extend(noise_events)
    
    return dataset


def save_dataset(dataset: List[Dict], filename: str = "golden_financial_reasoning.json"):
    """
    Save the dataset to a JSON file with pretty formatting.
    
    Args:
        dataset: List of market event dictionaries
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(dataset)} examples")
    print(f"✓ Saved to: {filename}")


def print_dataset_summary(dataset: List[Dict]):
    """
    Print a summary of the generated dataset.
    
    Args:
        dataset: List of market event dictionaries
    """
    print("\n" + "=" * 80)
    print("GOLDEN FINANCIAL REASONING DATASET SUMMARY")
    print("=" * 80)
    
    # Count by category
    categories = {}
    directions = {}
    volatilities = {}
    
    for item in dataset:
        cat = item['event_category']
        categories[cat] = categories.get(cat, 0) + 1
        
        direction = item['expected_direction']
        directions[direction] = directions.get(direction, 0) + 1
        
        vol = item['input_volatility']
        volatilities[vol] = volatilities.get(vol, 0) + 1
    
    print(f"\nTotal Examples: {len(dataset)}")
    
    print("\nBy Event Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:25s}: {count}")
    
    print("\nBy Expected Direction:")
    for direction, count in sorted(directions.items()):
        print(f"  {direction:25s}: {count}")
    
    print("\nBy Volatility Level:")
    for vol, count in sorted(volatilities.items()):
        print(f"  {vol:25s}: {count}")
    
    print("\n" + "=" * 80)
    
    # Show sample examples
    print("\nSAMPLE EXAMPLES:")
    print("=" * 80)
    
    for i, item in enumerate(dataset[:3], 1):
        print(f"\nExample {i} [{item['event_category']}]:")
        print(f"  News: {item['input_news']}")
        print(f"  Direction: {item['expected_direction']}")
        print(f"  Volatility: {item['input_volatility']}")
        print(f"  Reasoning: {', '.join(item['key_reasoning_points'][:2])}...")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """
    Main execution: Generate and save the golden dataset.
    """
    print("=" * 80)
    print("GENERATING GOLDEN FINANCIAL REASONING DATASET")
    print("=" * 80)
    
    # Generate dataset
    dataset = generate_golden_dataset()
    
    # Save to JSON
    save_dataset(dataset)
    
    # Print summary
    print_dataset_summary(dataset)
    
    print("\n✓ Dataset generation completed successfully!")
    print("=" * 80)
