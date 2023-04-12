from argparse import ArgumentParser
import tools as tl
import tools2 as tl2

parser = ArgumentParser()
parser.add_argument("--f_in", help = 'fichier d\'entrée de contexte de citations en format .csv.', required=True)
parser.add_argument("--f_out", help = 'fichier de sortie. If only want to plot, write none', required=True)
parser.add_argument("--max_size", help = 'taille max de nombre de tokens dans une entrée de bert')
parser.add_argument("--model", help='taper 1 pour bert et 0 pour sci-bert')
parser.add_argument("--mode", help='The comparision mode, all or modes separated by ,')
parser.add_argument("--plot", help='1: do a plot, 0: no plot')


if __name__ == "__main__":
    # Traitement des args
    args = parser.parse_args()
    f_in = args.f_in
    f_out = args.f_out
    max_size = args.max_size
    model = args.model
    plot = args.plot
    mode = args.mode

    max_size = 128 if args.max_size == None else int(args.max_size)
    mode = 'all' if args.mode == None else args.mode
    model = '1' if args.model == None else args.model

    """dic = tl.make_input(txt_path)
    with open('abstracts.txt', 'r', encoding='utf-8') as abs_in:
        abs = tl.get_cited_abs(cited_name, abs_in)
        title = tl.get_cited_title(cited_name, abs_in)

    if mode == 'mono':
        with open(f_out, 'w', encoding='utf-8') as f_out:
            tl.calculate_simi(dic, abs, max_size, f_out)
    elif mode == 'pair':
        with open(f_out, 'w', encoding='utf-8') as f_out:
            tl.calculate_simi_pair(dic, abs, title, max_size, f_out)"""

    if (f_out == 'none'): # only plot
        with open(f_in, 'r', encoding='utf-8') as f_out:
            tl2.pairplot(f_in, mode)
    

    df = tl2.make_input(f_in)

    if mode != 'all':
        compare = mode.split(',')
        compare1 = compare[0]
        compare2 = compare[1]
        with open(f_out, 'w', encoding='utf-8') as simi_out:
            tl2.calculate_simi(df, compare1, compare2, max_size, simi_out, model)
        
        if (plot == '1'):
            with open(f_out, 'r', encoding='utf-8') as f_out:
                tl2.histplot(f_out, mode)
    else:
        with open(f_out, 'w', encoding='utf-8') as simi_out:
            tl2.calculate_simi_all(df, max_size, simi_out, model)

        if (plot == '1'):
            with open(f_out, 'r', encoding='utf-8') as f_out:
                tl2.pairplot(f_out, model)
    
