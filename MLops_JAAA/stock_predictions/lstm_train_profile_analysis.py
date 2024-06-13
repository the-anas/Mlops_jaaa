import pstats

def print_profile_stats(profile_file):
    profile = pstats.Stats(profile_file)
    profile.strip_dirs().sort_stats('cumulative').print_stats(10)

if __name__ == "__main__":
    print_profile_stats('lstm_train_profile.prof')