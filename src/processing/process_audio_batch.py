#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from transcribe_to_pdf import transcribe_audio, create_pdf
from tqdm import tqdm

def process_audio_book(audio_path, output_dir, pdf_name):
    """Process a single audio book with progress tracking."""
    print(f"\nProcessing: {os.path.basename(audio_path)}")
    
    # Create output directory for transcription
    os.makedirs(output_dir, exist_ok=True)
    
    # Transcribe audio
    print("Transcribing audio...")
    text_file = transcribe_audio(audio_path, output_dir)
    
    # Create PDF in books directory
    print("Creating PDF...")
    pdf_file = create_pdf(text_file, "/Volumes/NVME_Expansion/musartao/data/books")
    
    # Rename PDF to desired name
    new_pdf_path = os.path.join("/Volumes/NVME_Expansion/musartao/data/books", f"{pdf_name}.pdf")
    os.rename(pdf_file, new_pdf_path)
    
    print(f"Completed: {pdf_name}")
    return True

def main():
    # Define batches of books to process
    batches = [
        # Batch 1: Psychology and Jungian works
        [
            ("AnimalSpiritsHowHumanPsychologyDrivestheEconomyandWhyItMattersforGlobalCapitalism_ep6.m4b", "AnimalSpirits"),
            ("PsychologyoftheUnconscious_ep6.m4b", "PsychologyOfTheUnconscious"),
            ("TheNeuroticCharacterFundamentalsofaComparativeIndividualPsychologyandPsychotherapy_ep7.m4b", "TheNeuroticCharacter"),
            ("CollectedPapersonAnalyticalPsychology_ep6.m4b", "CollectedPapersOnAnalyticalPsychology"),
            ("OntheNatureofthePsyche_ep7.m4b", "OnTheNatureOfThePsyche")
        ],
        # Batch 2: Philosophy and Science
        [
            ("PhishingforPhoolsTheEconomicsofManipulationandDeception_ep6.m4b", "PhishingForPhools"),
            ("TaoTeChingTheEssentialTranslationoftheAncientChineseBookoftheTao_ep6.m4b", "TaoTeChing"),
            ("AnswertoJob_ep7.m4b", "AnswerToJob"),
            ("ABriefHistoryofTime_ep6.aax", "ABriefHistoryOfTime"),
            ("TheGrandDesign_ep6.aax", "TheGrandDesign")
        ],
        # Batch 3: Design and Innovation
        [
            ("TheDesignofEverydayThings_ep6.aax", "TheDesignOfEverydayThings"),
            ("ChangebyDesignHowDesignThinkingTransformsOrganizationsandInspiresInnovation_ep6.aax", "ChangeByDesign"),
            ("CreativeConfidenceUnleashingtheCreativePotentialWithinUsAll_ep6.aax", "CreativeConfidence"),
            ("DesigningYourLifeHowtoBuildaWell-LivedJoyfulLife_ep6.aax", "DesigningYourLife")
        ],
        # Batch 4: Business and Economics
        [
            ("TheLeanStartupHowTodaysEntrepreneursUseContinuousInnovationtoCreateRadicallySuccessfulB_ep6.aax", "TheLeanStartup"),
            ("TheInnovatorsDilemmaMeetingtheChallengeofDisruptiveChange_ep6.aax", "TheInnovatorsDilemma"),
            ("CrossingtheChasmMarketingandSellingTechnologyProjectstoMainstreamCustomers_ep6.aax", "CrossingTheChasm"),
            ("TheBlackSwanTheImpactoftheHighlyImprobable_ep6.aax", "TheBlackSwan")
        ],
        # Batch 5: Science and Technology
        [
            ("TheSelfishGene_ep6.aax", "TheSelfishGene"),
            ("WhyWeSleepUnlockingthePowerofSleepandDreams_ep6.aax", "WhyWeSleep"),
            ("TheAgeofSpiritualMachinesWhenComputersExceedHumanIntelligence_ep6.aax", "TheAgeOfSpiritualMachines"),
            ("HowtoCreateaMindTheSecretofHumanThoughtRevealed_ep6.aax", "HowToCreateAMind")
        ]
    ]
    
    parser = argparse.ArgumentParser(description="Process audio books in batches")
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4, 5], help="Which batch to process (1-5)")
    parser.add_argument("--all", action="store_true", help="Process all batches")
    args = parser.parse_args()
    
    if not args.batch and not args.all:
        print("Please specify either --batch <number> or --all")
        return
    
    audio_dir = "/Volumes/NVME_Expansion/musartao/data/books/AudioIngest"
    output_dir = "/Volumes/NVME_Expansion/musartao/data/books/AudioIngest/transcriptions"
    
    # Process specified batches
    if args.all:
        batches_to_process = batches
    else:
        batches_to_process = [batches[args.batch - 1]]
    
    for batch_num, batch in enumerate(batches_to_process, 1):
        print(f"\nProcessing Batch {batch_num}...")
        for audio_file, pdf_name in batch:
            audio_path = os.path.join(audio_dir, audio_file)
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_file}")
                continue
            process_audio_book(audio_path, output_dir, pdf_name)
        
        if batch_num < len(batches_to_process):
            print("\nBatch completed. Press Enter to continue to next batch, or Ctrl+C to stop...")
            input()

if __name__ == "__main__":
    main() 