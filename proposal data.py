import matplotlib.pyplot as plt
import numpy as np

# Sample data
datasets = [
    "IMDB-WIKI",
    "[X] Human Object Interaction Processing (HOIP)",
    "The Asian Face Age Dataset",
    "Cross-Age Celebrity Dataset",
    "WebFace",
    "[X] MORPH",
    "Specs on Face (SoF)",
    "MegaAge",
    "Adience",
    "UTKFace",
    "AgeDB",
    "MSU LFW+",
    "Facial Recognition Technology (FERET)",
    "[X] YGA",
    "Image of Group",
    "[X] Iranian Face Database",
    "[X] FG-NET"
    ]

sample_counts = [
    460723,
    306600,
    164432,
    163446,
    494414,
    55134,
    42592,
    41941,
    26580,
    23000,
    16488,
    15999,
    14126,
    8000,
    5080,
    3600,
    1002
    ]

wiki_sample = [62328, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


datasets.reverse()
sample_counts.reverse()
wiki_sample.reverse()

# Create the bars
width = 0.35
x = np.arange(len(datasets))

plt.barh(x, sample_counts, width)
plt.barh(x, wiki_sample, width, left=sample_counts, color='gray', label='WIKI')

# Add labels and title
plt.xlabel('Samples')
plt.yticks(x, datasets)
plt.legend(loc='lower right')

# Display the plot
plt.show()