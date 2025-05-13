from langchain.text_splitter import CharacterTextSplitter

text =""" Hummingbirds migrate north in the spring, so if you want to make your garden a destination for them, you'll need to grow nectar-rich flowers. Opting for container plants is especially beneficial, as you can place them on elevated surfaces, creating more protection against these birds' natural predators.

It's a good idea to bring hummingbirds to your garden, not just because they're beautiful, but because they're beneficial pollinators and can even consume troublesome insects (like mosquitoes). Here are 10 container plants that experts say will attract these special birds.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)
print(result)