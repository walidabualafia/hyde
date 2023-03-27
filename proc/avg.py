import os

word = 'islam'

# define function to read in numbers from a file and return their average
def average_numbers(file_path):
    with open(file_path, 'r') as file:
        numbers = file.readlines()
        numbers = [float(num.strip()) for num in numbers]
        return sum(numbers) / len(numbers)

# define function to get all text files in a directory
def get_text_files(directory):
    text_files = []
    for folder in os.listdir(directory):
        print(folder)
        for file in os.listdir(f'../out/{word}/{folder}/'):
            if not file.startswith(f'{word}'):
                text_files.append(os.path.join(directory, f'{folder}/{file}'))
                print(f'    {file}')
    return text_files

# get all text files in the current directory
text_files = get_text_files(f'../out/{word}/')

# calculate the average for each text file
averages = []
for file in text_files:
    avg = average_numbers(file)
    averages.append(avg)

# write the averages to a new file
with open('averages.txt', 'w') as file:
    for i in range(len(averages)):
        if text_files[i].find('positive') != -1:
            file.write(text_files[i][text_files[i].index('positive'):] + '\n' + str(averages[i]) + '\n\n')
        elif text_files[i].find('neutral') != -1:
            file.write(text_files[i][text_files[i].index('neutral'):] + '\n' + str(averages[i]) + '\n\n')
        elif text_files[i].find('negative') != -1:
            file.write(text_files[i][text_files[i].index('negative'):] + '\n' + str(averages[i]) + '\n\n')

print("Averages written to averages.txt")
print(f'../out/{word}')
print(get_text_files(f'../out/{word}'))

