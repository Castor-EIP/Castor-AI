import argparse

def main():
    parser = argparse.ArgumentParser(description="Castor AI launcher")
    parser.add_argument("--mode", choices=["sport", "podcast"], required=True)

    # Arguments sport-ai (optionnels)
    parser.add_argument("--video-left", type=str, help="Chemin video gauche (sport)")
    parser.add_argument("--video-right", type=str, help="Chemin video droite (sport)")
    parser.add_argument("--frameskip", type=int, default=0, help="Frameskip (sport)")

    args = parser.parse_args()

    if args.mode == "sport":
        from modules.sport_ai.inference import run as run_sport
        if not args.video_left or not args.video_right:
            parser.error("--video-left et --video-right sont requis en mode sport")
        run_sport(args.video_left, args.video_right, frameskip=args.frameskip)

    elif args.mode == "podcast":
        #from modules.podcast_ai.inference import run as run_podcast
        #run_podcast()
        print("Mode podcast non encore implémenté.")

if __name__ == "__main__":
    main()