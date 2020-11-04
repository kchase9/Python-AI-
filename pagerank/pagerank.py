import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    item_probability = dict()
    for var in corpus:
        item_probability[var] = (1 - damping_factor)/len(corpus)


    if corpus[page]:
        for item in corpus[page]:
            item_probability[item] += damping_factor/len(corpus[page])
    else:
        for item in corpus:
            item_probability[item] = 1/len(corpus)


    return item_probability

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #declare the dictionary you're going to return
    sample_pr = dict()

    #initialize and store in the sample dictionary the values of the pages in the corpus
    for item in corpus:
        sample_pr[item] = float(0)

    #select random page
    sample_page = random.choices(list(corpus.keys()))[0]

    #a page has already been selected
    for var in range(1, n):
        #store the transition model dictionary
        trans_model = transition_model(corpus, sample_page, damping_factor)
        for page in corpus:
            #adjust the pagerank
            sample_pr[page] = ((var - 1) * sample_pr[page] + trans_model[page])/var

        #select a new random page
        sample_page = random.choices(list(corpus.keys()), list(sample_pr.values()), k=1)[0]

    #normalize the results
    sample_sum = 0
    n = sum(sample_pr.values())
    for page in sample_pr:
        sample_pr[page] /= n
        sample_sum += sample_pr[page]
    print("Sample Pagerank Sum:", sample_sum)

    return sample_pr


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict().fromkeys(list(corpus.keys()), 1/len(corpus))
    new_pagerank = dict()

    var = True
    #count changes when the difference in pr values changes by less than the min
    while var:
        count = 0
        for page in pagerank:
            total_pagerank = float(0)
            num_links = 0

            #loop through each link in this page
            #add to total pagerank PR(link)/NumLinks(link)
            #multiply total pagerank by the damping factor
            #check if the page has no links and adjust properly

            for link in corpus:
                if page in corpus[link]:
                    total_pagerank += pagerank[link]/len(corpus[link])
                    num_links += 1

            if num_links == 0:
                for link in corpus:
                    total_pagerank += pagerank[link]/len(corpus)

            new_pagerank[page] = (1 - damping_factor)/len(corpus) + total_pagerank * damping_factor

        #check if the values change by more than 0.001
        for page in pagerank:
            if abs(pagerank[page]-new_pagerank[page]) <= 0.005:
                count += 1

        for page in pagerank:
            pagerank[page] = new_pagerank[page]

        if count == len(corpus):
            var = False


    sum0 = sum(pagerank.values())
    print("\nIteration Pagerank Sum:", sum0)

    return pagerank


if __name__ == "__main__":
    main()
