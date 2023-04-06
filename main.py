from argparse import ArgumentParser
import tools as tl

parser = ArgumentParser()
parser.add_argument("--txt_path", help = 'fichier d\'entrée de contexte de citations en format .txt, utf-8')
parser.add_argument("--f_out", help = 'fichier de sortie')
parser.add_argument("--max_size", help = 'taille max de nombre de tokens dans une entrée de bert')
parser.add_argument("--model", help='taper 1 ou 2 ou 3 pour choisir un model de bert')
parser.add_argument("--cited_name", help='The name of the cited paper author')
parser.add_argument("--plot", help='1: do a plot, 0: no plot')

if __name__ == "__main__":
    # Traitement des args
    args = parser.parse_args()
    txt_path = args.txt_path
    f_out = args.f_out
    max_size = args.max_size
    model = args.model
    cited_name = args.cited_name
    plot = args.plot

    max_size = 128 if args.max_size == None else int(args.max_size)

    dic = tl.make_input(txt_path)
    with open('abstracts.txt', 'r', encoding='utf-8') as abs_in:
        abs = tl.get_cited_abs(cited_name, abs_in)

    with open(f_out, 'w', encoding='utf-8') as f_out:
        tl.calculate_simi(dic, abs, max_size, f_out)
    if (plot == '1'):
        with open(f_out, 'r', encoding='utf-8') as f_out:
            tl.histplot(f_out)