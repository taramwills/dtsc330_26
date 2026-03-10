import pandas as pd
import random

def random_zip_code():
    return f"{random.randint(0, 99999):05d}"

first_names = ["Aiden", "Maya", "Lucas", "Zoe", "Ethan", "Nina", "Caleb", "Lila", "Jordan", "Sofia"]

last_names = ["Hawkins", "Ramirez", "Bennett", "Coleman", "Nguyen", "Patel", "Olsen", "Thompson", "Morales", "Kowalski"]

data = []

people = 100

for person in range(people):
    data.append({"forename" : random.choice(first_names),
                 "surname" : random.choice(last_names),
                 "zipcode" : random_zip_code()})
    
df = pd.DataFrame(data)

print(df)