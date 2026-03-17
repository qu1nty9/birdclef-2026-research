# BirdCLEF+ 2026 Competition Overview

## Overview

The goal of this competition is to develop machine learning frameworks capable of identifying understudied species within continuous audio data from Brazil's Pantanal wetlands. Successful solutions will help advance biodiversity monitoring in the last wild places on Earth.

## Description

How do you protect an ecosystem you can’t fully see? One way is to listen.

This competition involves building models that automatically identify wildlife species from their vocalizations in audio recordings collected across the Pantanal wetlands. This work will support more reliable biodiversity monitoring in one of the world’s most diverse and threatened ecosystems.

Understanding how ecological communities respond to environmental change and restoration efforts is a central challenge in conservation science. The Pantanal — a wetland spanning 150,000+ km² across Brazil and neighboring countries — is home to over 650 bird species plus countless other animals, yet much of it remains unmonitored. Seasonal flooding, wildfires, agricultural expansion, and climate change make regular fieldwork challenging.

## Goal of the Competition

Conventional biodiversity monitoring across vast, remote regions is expensive and logistically demanding. To help address these challenges, a growing network of 1,000 acoustic recorders is being deployed across the Pantanal, running continuously to capture wildlife sounds across different habitats and seasons. Continuous audio recording allows researchers to capture multi-species soundscapes over extended periods, providing a community-level perspective on biodiversity dynamics. But the sheer volume of audio is too large to review manually, and labeled species data is limited.

This competition focuses on the development of machine learning models that identify wildlife species from passive acoustic monitoring (PAM). Proposed approaches should work across different habitats, withstand the constraints of messy, field-collected data, and support evidence-based conservation decisions. Successful solutions will help advance biodiversity monitoring in the last wild places on Earth, including research initiatives in the Pantanal wetlands of Brazil.

Listening carefully, and at scale, may be one of the most effective tools available to protect this landscape.

## Timeline

 - March 11, 2026 - Start Date.

 - May 27, 2026 - Entry Deadline. You must accept the competition rules before this date to compete.

 - May 27, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.

 - June 3, 2026 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Submission Format

For each row_id, you should predict the probability that a given species was present. There is one column per species. Each row covers a five-second window of audio.

## Code Requirements

This is a Code Competition

Submissions to this competition must be made through Notebooks. For the "Submit" button to be active after a commit, the following conditions must be met:

 - CPU Notebook <= 90 minutes run-time
 - GPU Notebook submissions are disabled. You can technically submit but will only have 1 minute of runtime.
 - Internet access disabled
 - Freely & publicly available external data is allowed, including pre-trained models
 - Submission file must be named submission.csv
 
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you encounter submission errors.