alice = ["Tablet", "Laptop", "Headphones", "Mouse", "Keyboard", "Laptop", "Headphones", "Headphones", "Mouse", "Tablet"]
bob = ["Mouse", "Laptop", "Tablet", "Laptop", "Keyboard"]
carol = ["Tablet", "Laptop", "Keyboard", "Headphones", "Headphones"]

product = ["Laptop", "Tablet", "Headphones", "Keyboard", "Mouse"]

def encode(purchase_history, products):
    encoded_vector = []
    
    for item in purchase_history:
        if item in products:
            product_index = products.index(item) + 1
            encoded_vector.append(product_index)
            
    return encoded_vector

def sliding_window(data, window_size):
    behavior_vectors = []
    
    for i in range(len(data) - window_size + 1):
        behavior_vector = data[i:i + window_size]
        behavior_vectors.append(behavior_vector)
    
    return behavior_vectors

def cosine_sim(A, B):
    magnitudeA = np.linalg.norm(A)
    magnitudeB = np.linalg.norm(B)

    return np.dot(A, B) / (magnitudeA * magnitudeB)

def cosine_list(name_people, base):
    base_slidings = sliding_window(base, window_size= len(name_people))
    list_cosine =[]
    for base_sliding in base_slidings:
        #print(base_sliding)
        list_cosine.append(round(cosine_sim(name_people, base_sliding),4))
        if(len(list_cosine)==len(name_people)):
            break
    return list_cosine

def recommend(cosine_list, name_people, base):
    index_max_cosine = cosine_list.index(max(cosine_list))
    return base[index_max_cosine + len(name_people)]

alice_encoded = encode(alice, product)
print(alice_encoded)
bob_encoded = encode(bob, product)
print(bob_encoded)
carol_encoded = encode(carol, product)

bob_cosine = cosine_list(bob_encoded, alice_encoded)
carol_cosine = cosine_list(carol_encoded, alice_encoded)
print(f"Cosine similarity of Bob’s behavior and each sliding window of Alice’s behavior:\n{bob_cosine}")
print(f"Cosine similarity of Carol's behavior and each sliding window of Alice’s behavior:\n{carol_cosine}")

print(recommend(bob_cosine, bob, alice))
print(recommend(carol_cosine, carol, alice))
