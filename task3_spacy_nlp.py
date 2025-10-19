# task3_nltk_alternative.py - Alternative using NLTK instead of spaCy
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    print("✓ NLTK data downloaded successfully")
except:
    print("⚠️  NLTK download issues, but continuing...")

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

# Sample Amazon reviews data (same as before)
reviews = [
    "I absolutely love my new iPhone 14 Pro from Apple. The camera quality is amazing!",
    "Samsung Galaxy S23 has terrible battery life. Very disappointed with Samsung.",
    "The Sony WH-1000XM4 headphones have excellent noise cancellation and sound quality.",
    "Amazon Echo Dot is okay, but Alexa doesn't understand my commands sometimes.",
    "Microsoft Surface Laptop 5 is fantastic for work and gaming applications.",
    "The Google Pixel 7 camera is incredible, but battery could be better.",
    "Bose QuietComfort headphones are worth every penny for the comfort and audio quality.",
    "This HP laptop keeps crashing and has the worst customer service experience.",
    "Apple MacBook Pro with M2 chip is lightning fast and reliable for programming.",
    "Dell XPS 13 is a beautiful machine but has some heating issues during heavy use.",
    "I'm very happy with my new iPad from Apple, the display is gorgeous!",
    "The battery life on this Samsung phone is terrible, I don't recommend it.",
    "Microsoft Windows 11 runs smoothly on my new computer, very impressed.",
    "This product from Google is amazing, but the price is too high for me.",
    "The sound quality of these Sony speakers is absolutely incredible!"
]

# Known brands and products for entity recognition
known_brands = {
    'apple', 'samsung', 'sony', 'microsoft', 'google', 'bose', 'hp', 'dell', 'amazon'
}

known_products = {
    'iphone', 'ipad', 'macbook', 'galaxy', 'surface', 'pixel', 'echo', 'windows',
    'laptop', 'phone', 'headphones', 'speakers', 'computer'
}

# Sentiment lexicon
positive_words = {
    'love', 'amazing', 'excellent', 'fantastic', 'incredible', 'great', 
    'good', 'awesome', 'wonderful', 'best', 'fast', 'reliable', 'beautiful', 
    'worth', 'lightning', 'happy', 'gorgeous', 'smoothly', 'impressed'
}

negative_words = {
    'terrible', 'disappointed', 'worst', 'crashing', 'issues', 'bad', 
    'poor', 'horrible', 'awful', 'problem', 'broken'
}

def extract_entities_nltk(text):
    """Extract entities using NLTK"""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Named Entity Recognition
    tree = ne_chunk(pos_tags)
    entities = []
    
    # Convert tree to IOB format and extract entities
    iob_tags = tree2conlltags(tree)
    
    current_entity = []
    current_label = None
    
    for token, pos, label in iob_tags:
        if label != 'O':
            if label.startswith('B-'):
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = [token]
                current_label = label[2:]
            elif label.startswith('I-'):
                current_entity.append(token)
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_label))
                current_entity = []
                current_label = None
    
    if current_entity:
        entities.append((' '.join(current_entity), current_label))
    
    return entities

def rule_based_entity_recognition(text):
    """Rule-based entity recognition as fallback"""
    tokens = text.lower().split()
    brands_found = []
    products_found = []
    
    # Check for known brands and products
    for i, token in enumerate(tokens):
        # Clean token
        clean_token = token.strip('.,!?;:')
        
        if clean_token in known_brands:
            brands_found.append(clean_token.title())
        
        if clean_token in known_products:
            products_found.append(clean_token.title())
        
        # Check for multi-word entities
        if i < len(tokens) - 1:
            two_word = f"{clean_token} {tokens[i+1].strip('.,!?;:')}"
            if two_word in known_brands:
                brands_found.append(two_word.title())
            if two_word in known_products:
                products_found.append(two_word.title())
    
    return brands_found, products_found

def analyze_sentiment(text):
    """Rule-based sentiment analysis"""
    tokens = text.lower().split()
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

print("=" * 80)
print("NLP Analysis with NLTK (Alternative to spaCy)")
print("=" * 80)

products_mentioned = []
brands_mentioned = []
sentiment_results = []

for i, review in enumerate(reviews, 1):
    sentiment = analyze_sentiment(review)
    sentiment_results.append(sentiment)
    
    # Try NLTK NER first, then fallback to rule-based
    try:
        entities = extract_entities_nltk(review)
        brands = [ent[0] for ent in entities if ent[1] in ['ORGANIZATION', 'PERSON', 'GPE']]
        products = [ent[0] for ent in entities if ent[1] in ['PRODUCT', 'ORGANIZATION']]
    except:
        brands, products = [], []
    
    # Fallback to rule-based recognition
    if not brands or not products:
        rule_brands, rule_products = rule_based_entity_recognition(review)
        brands.extend(rule_brands)
        products.extend(rule_products)
    
    brands_mentioned.extend(brands)
    products_mentioned.extend(products)
    
    print(f"\nReview {i}: {review}")
    print(f"Sentiment: {sentiment.upper()}")
    print(f"Brands mentioned: {brands}")
    print(f"Products mentioned: {products}")
    print("-" * 80)

# Analyze overall results
sentiment_counts = Counter(sentiment_results)
product_counts = Counter(products_mentioned)
brand_counts = Counter(brands_mentioned)

print("\n" + "=" * 80)
print("OVERALL ANALYSIS RESULTS")
print("=" * 80)
print(f"Sentiment Distribution: {dict(sentiment_counts)}")
print(f"Most mentioned products: {product_counts.most_common(10)}")
print(f"Most mentioned brands: {brand_counts.most_common(10)}")

# Create visualizations
plt.figure(figsize=(15, 5))

# Sentiment distribution
plt.subplot(1, 3, 1)
colors = ['#4CAF50', '#F44336', '#FFC107']
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors)
plt.title('Sentiment Distribution')
plt.ylabel('Number of Reviews')

# Top products mentioned
plt.subplot(1, 3, 2)
top_products = product_counts.most_common(5)
if top_products:
    products, counts = zip(*top_products)
    plt.bar(products, counts, color='skyblue')
    plt.title('Top 5 Products Mentioned')
    plt.xticks(rotation=45, ha='right')

# Top brands mentioned
plt.subplot(1, 3, 3)
top_brands = brand_counts.most_common(5)
if top_brands:
    brands, counts = zip(*top_brands)
    plt.bar(brands, counts, color='lightcoral')
    plt.title('Top 5 Brands Mentioned')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('images/nltk_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Analysis complete using NLTK!")
print(f"  - Processed {len(reviews)} reviews")
print(f"  - Found {len(products_mentioned)} product mentions")
print(f"  - Found {len(brands_mentioned)} brand mentions")
print(f"  - Visualization saved to: images/nltk_analysis_results.png")