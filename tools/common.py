import sys

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    progress = 100 * (iteration / float(total))
    percent = ("{0:." + str(decimals) + "f}").format(progress)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('%s |%s| %s%% %s \r' % (prefix, bar, percent, suffix))

    if progress == 100.:
        sys.stdout.write('\n')
    sys.stdout.flush()

