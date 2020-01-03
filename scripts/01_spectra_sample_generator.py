from srsmp import *


def input_string(prompt_text):
    user_input = input(prompt_text)
    text = user_input.replace(" ","_")
    if not text: # Can be made more robust
        print("Empty strings not allowed!")
        text = input_string(prompt_text)
    return text

def input_int(prompt_text, default=None):
    if default is not None:
        user_input = input(prompt_text) or default
    else:
        user_input = input(prompt_text)
    try:
        val = int(user_input)
    except ValueError:
        print("Input must be an integer!")
        val = input_int(prompt_text, default)
    return val


if __name__ == "__main__":
    input("Make sure all settings are correct in 'config.ini' and press ENTER to continue.")
    
    class_id = input_string("Provide class identifier (avoid whitespaces): ")
    campaign_nr = input_int("Enter campaign number or -1 for automatic increase [-1]: ", -1)
    spg = SpectraGenerator(class_id, campaign_nr)

    sample_size = input_int("Number of samples to be taken [100]: ", 100)


    spg.measure_spectra(sample_size)


