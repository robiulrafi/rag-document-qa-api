import sqlite3

conn = sqlite3.connect("chroma_db/chroma.sqlite3")
cur = conn.cursor()

# What tables exist?
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in cur.fetchall()])

# The original text you stored lives in embedding_metadata
print("\n--- Stored documents ---")
cur.execute("SELECT string_value FROM embedding_metadata WHERE key='chroma:document'")
for row in cur.fetchall():
    print(" •", row[0])

# How many embeddings are stored?
cur.execute("SELECT COUNT(*) FROM embeddings")
print("\nEmbedding count:", cur.fetchone()[0])

conn.close()