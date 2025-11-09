# main.py
import argparse
#from modules.sport-ai.inference import SportAI
#from modules.podcast-ai.inference import PodcastAI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sport", "podcast"], required=True)
    args = parser.parse_args()

#    if args.mode == "sport":
#        ai = SportAI()
#    elif args.mode == "podcast":
#        ai = PodcastAI()

#    ai.run()

if __name__ == "__main__":
    main()
