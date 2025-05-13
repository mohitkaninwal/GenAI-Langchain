from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text ="""
# Python program to swap two variables

x = 5
y = 10

# To take inputs from the user
#x = input('Enter value of x: ')
#y = input('Enter value of y: ')

# create a temporary variable and swap the values
temp = x
x = y
y = temp

print('The value of x after swapping: {}'.format(x))
print('The value of y after swapping: {}'.format(y))

"""

splliter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=52,
    chunk_overlap=0
)

chunks = splliter.split_text(text)
print(chunks[0])