from PIL import Image
from math import log

def entropy(img):
    """
    img : PIL Image
    """
    histogram = img.histogram()
    log2 = lambda x:log(x)/log(2)

    total = len(histogram)
    counts = {}
    for item in histogram:
        counts.setdefault(item,0)
        counts[item]+=1

    ent = 0
    for i in counts:
        p = float(counts[i])/total
        ent-=p*log2(p)
    return -ent*log2(1/ent)


if __name__ == "__main__":
    im = Image.open('confusion_matrix.png').convert('L')
    print entropy(im)