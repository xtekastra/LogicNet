import copy
import torch
import asyncio
import threading
import bittensor as bt

from typing import List
from traceback import print_exception

from logicnet.base.neuron import BaseNeuron


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"\033[1;32m🔗 Dendrite: {self.dendrite}\033[0m")

        # Set up initial scoring weights for validation
        bt.logging.info("\033[1;32m⚖️ Building validation weights.\033[0m")
        self.scores = torch.zeros_like(self.metagraph.S.clone().detach(), dtype=torch.float32)

        # Init sync with the network. Updates the metagraph.
        self.resync_metagraph()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("\033[1;33m⚠️ axon off, not serving ip to chain.\033[0m")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("\033[1;32m🌐 serving ip to chain...\033[0m")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"\033[1;31m❌ Failed to serve Axon with exception: {e}\033[0m")
                pass

        except Exception as e:
            bt.logging.error(f"\033[1;31m❌ Failed to create Axon initialize with exception: {e}\033[0m")
            pass

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"\033[1;32m🧠 Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}\033[0m"
        )
        if hasattr(self, "axon"):
            f"Axon: {self.axon}"

        bt.logging.info(f"\033[1;32m🧱 Validator starting at block: {self.block}\033[0m")

        # This loop maintains the validator's operations until intentionally stopped.
        while True:
            try:
                if self.config.proxy.port:
                    try:
                        self.validator_proxy.get_credentials()
                        bt.logging.info(
                            "\033[1;32m🔌 Validator proxy ping to proxy-client successfully\033[0m"
                        )
                    except Exception:
                        bt.logging.warning("\033[1;33m⚠️ Warning, proxy can't ping to proxy-client.\033[0m")

                bt.logging.info(f"\033[1;32m🔄 step({self.step}) block({self.block})\033[0m")

                # Run forward.
                try:
                    self.forward()
                    bt.logging.info(f"\033[1;32m🔄 step({self.step}) block({self.block}) forward() completed\033[0m")
                except Exception as err:
                    print_exception(type(err), err, err.__traceback__)

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.save_state()

                self.step += 1

            # If someone intentionally stops the validator, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("\033[1;32m🛑 Validator killed by keyboard interrupt.\033[0m")
                exit()

            # In case of unforeseen errors, the validator will log the error and continue operations.
            except Exception as err:
                bt.logging.error("\033[1;31m❌ Error during validation\033[0m", str(err))
                bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("\033[1;32m🚀 Starting validator in background thread.\033[0m")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("\033[1;32m✅ Started\033[0m")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("\033[1;33m🛑 Stopping validator in background thread.\033[0m")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("\033[1;32m✅ Stopped\033[0m")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("\033[1;33m🛑 Stopping validator in background thread.\033[0m")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("\033[1;32m✅ Stopped\033[0m")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                "\033[1;33m⚠️ Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions.\033[0m"
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)
        bt.logging.info(f"raw_weights {raw_weights}")
        bt.logging.info(f"top10 values {raw_weights.sort()[0]}")
        bt.logging.info(f"top10 uids {raw_weights.sort()[1]}")

        # Convert uids to a PyTorch tensor before processing
        uids_tensor = self.metagraph.uids.clone().detach()

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=uids_tensor.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.info(f"processed_weights {processed_weights}")
        bt.logging.info(f"processed_weight_uids {processed_weight_uids}")

        # Set the weights on chain via our subtensor connection.
        self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=self.spec_version,
        )

        bt.logging.info(f"\033[1;32m⚖️ Set weights: {processed_weights}\033[0m")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("\033[1;32m🔄 resync_metagraph()\033[0m")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "\033[1;32m🔄 Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages\033[0m"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if (hotkey != self.metagraph.hotkeys[uid]):
                bt.logging.info(f"\033[1;32m🔄 Hotkey {hotkey} has been replaced\033[0m")
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            bt.logging.info(
                "\033[1;32m🔄 Metagraph has grown, adding new hotkeys and moving averages\033[0m"
            )
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"\033[1;33m⚠️ NaN values detected in rewards: {rewards}\033[0m")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(uids).to(self.device), rewards
        ).to(self.device)
        bt.logging.debug(f"\033[1;32m🔄 Scattered rewards: {rewards}\033[0m")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.info(f"\033[1;32m📈 Updated moving avg scores: {self.scores}\033[0m")
