hdfs_templates = [
    "Adding an already existing block (.*)",
    "(.*)Verification succeeded for (.*)",
    "(.*) Served block (.*) to (.*)",
    "(.*):Got exception while serving (.*) to (.*):(.*)",
    "Receiving block (.*) src: (.*) dest: (.*)",
    "Received block (.*) src: (.*) dest: (.*) of size ([-]?[0-9]+)",
    "writeBlock (.*) received exception (.*)",
    "PacketResponder ([-]?[0-9]+) for block (.*) Interrupted\.",
    "Received block (.*) of size ([-]?[0-9]+) from (.*)",
    "PacketResponder (.*) ([-]?[0-9]+) Exception (.*)",
    "PacketResponder ([-]?[0-9]+) for block (.*) terminating",
    "(.*):Exception writing block (.*) to mirror (.*)(.*)",
    "Receiving empty packet for block (.*)",
    "Exception in receiveBlock for block (.*) (.*)",
    "Changing block file offset of block (.*) from ([-]?[0-9]+) to ([-]?[0-9]+) meta file offset to ([-]?[0-9]+)",
    "(.*):Transmitted block (.*) to (.*)",
    "(.*):Failed to transfer (.*) to (.*) got (.*)",
    "(.*) Starting thread to transfer block (.*) to (.*)",
    "Reopen Block (.*)",
    "Unexpected error trying to delete block (.*)\. BlockInfo not found in volumeMap\.",
    "Deleting block (.*) file (.*)",
    "BLOCK\* NameSystem\.allocateBlock: (.*)\. (.*)",
    "BLOCK\* NameSystem\.delete: (.*) is added to invalidSet of (.*)",
    "BLOCK\* Removing block (.*) from neededReplications as it does not belong to any file\.",
    "BLOCK\* ask (.*) to replicate (.*) to (.*)",
    "BLOCK\* NameSystem\.addStoredBlock: blockMap updated: (.*) is added to (.*) size ([-]?[0-9]+)",
    "BLOCK\* NameSystem\.addStoredBlock: Redundant addStoredBlock request received for (.*) on (.*) size ([-]?[0-9]+)",
    "BLOCK\* NameSystem\.addStoredBlock: addStoredBlock request received for (.*) on (.*) size ([-]?[0-9]+) But it does not belong to any file\.",
    "PendingReplicationMonitor timed out block (.*)",
]

bgl_templates = [
    "(.*):(.*) (.*):(.*) (.*):(.*) (.*):(.*)",
    "(.*) (.*) (.*) BGLERR_IDO_PKT_TIMEOUT connection lost to nodelinkservice card",
    "(.*) correctable errors exceeds threshold iar (.*) lr (.*)",
    "(.*) ddr errors detected and corrected on rank (.*) symbol (.*) over (.*) seconds",
    "(.*) ddr errorss detected and corrected on rank (.*) symbol (.*), bit (.*)",
    "(.*) double-hummer alignment exceptions",
    "(.*) exited abnormally due to signal: Aborted",
    "(.*) exited normally with exit code (.*)",
    "(.*) floating point alignment exceptions",
    "(.*) L3 (.*) errors dcr (.*) detected and corrected over (.*) seconds",
    "(.*) L3 (.*) errors dcr (.*) detected and corrected",
    "(.*) microseconds spent in the rbs signal handler during (.*) calls (.*) microseconds was the maximum time for a single instance of a correctable ddr",
    "(.*) PGOOD error latched on link card",
    "(.*) power module (.*) is not accessible",
    "(.*) TLB error interrupt",
    "(.*) torus non-crc errors dcr (.*) detected and corrected over (.*) seconds",
    "(.*) torus non-crc errors dcr (.*) detected and corrected",
    "(.*) torus receiver (.*) input pipe errors dcr (.*) detected and corrected",
    "(.*) torus receiver (.*) input pipe errors dcr (.*) detected and corrected over (.*) seconds",
    "(.*) torus (.*) (.*) (.*) errors dcr (.*) detected and corrected",
    "(.*) torus (.*) (.*) (.*) errors dcr (.*) detected and corrected over (.*) seconds",
    "(.*) total interrupts (.*) critical input interrupts (.*) microseconds total spent on critical input interrupts, (.*) microseconds max time in a critical input interrupt",
    "(.*) tree receiver (.*) in re-synch state events dcr (.*) detected",
    "(.*) tree receiver (.*) in re-synch state events dcr (.*) detected over (.*) seconds",
    "Added (.*) subnets and (.*) addresses to DB",
    "address parity error0",
    "auxiliary processor0",
    "Bad cable going into LinkCard (.*) Jtag (.*) Port (.*) - (.*) bad wires",
    "BglIdoChip table has (.*) IDOs with the same IP address (.*)",
    "BGLMASTER FAILURE mmcs_server exited normally with exit code 13",
    "BGLMaster has been started: BGLMaster --consoleip 127001 --consoleport 32035 --configfile bglmasterinit",
    "BGLMaster has been started: BGLMaster --consoleip 127001 --consoleport 32035 --configfile bglmasterinit --autorestart y",
    "BGLMaster has been started: BGLMaster --consoleip 127001 --consoleport 32035 --configfile bglmasterinit --autorestart y --db2profile ubgdb2cli",
    "byte ordering exception0",
    "Can not get assembly information for node card",
    "capture (.*)",
    "capture first (.*) (.*) error address0",
    "CE sym (.*) at (.*) mask (.*)",
    "CHECK_INITIAL_GLOBAL_INTERRUPT_VALUES",
    "chip select0",
    "ciod: (.*) coordinate (.*) exceeds physical dimension (.*) at line (.*) of node map file (.*)",
    "ciod: cpu (.*) at treeaddr (.*) sent unrecognized message (.*)",
    "ciod: duplicate canonical-rank (.*) to logical-rank (.*) mapping at line (.*) of node map file (.*)",
    "ciod: Error creating node map from file (.*) Argument list too long",
    "ciod: Error creating node map from file (.*) Bad address",
    "ciod: Error creating node map from file (.*) Bad file descriptor",
    "ciod: Error creating node map from file (.*) Block device required",
    "ciod: Error creating node map from file (.*) Cannot allocate memory",
    "ciod: Error creating node map from file (.*) Device or resource busy",
    "ciod: Error creating node map from file (.*) No child processes",
    "ciod: Error creating node map from file (.*) No such file or directory",
    "ciod: Error creating node map from file (.*) Permission denied",
    "ciod: Error creating node map from file (.*) Resource temporarily unavailable",
    "ciod: Error loading (.*) invalid or missing program image, Exec format error",
    "ciod: Error loading (.*) invalid or missing program image, No such device",
    "ciod: Error loading (.*) invalid or missing program image, No such file or directory",
    "ciod: Error loading (.*) invalid or missing program image, Permission denied",
    "ciod: Error loading (.*) not a CNK program image",
    "ciod: Error loading (.*) program image too big, (.*) > (.*)",
    "ciod: Error loading -mode VN: invalid or missing program image, No such file or directory",
    "ciod: Error opening node map file (.*) No such file or directory",
    "ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to (.*) (.*) (.*) (.*) (.*)",
    "ciod: Error reading message prefix on CioStream socket to (.*) Connection reset by peer",
    "ciod: Error reading message prefix on CioStream socket to (.*) Connection timed out",
    "ciod: Error reading message prefix on CioStream socket to (.*) Link has been severed",
    "ciod: failed to read message prefix on control stream CioStream socket to (.*)",
    "ciod: for node (.*) incomplete data written to core file (.*)",
    "ciod: for node (.*) read continuation request but ioState is (.*)",
    "ciod: generated (.*) core files for program (.*)",
    "ciod: In packet from node (.*) (.*) message code (.*) is not (.*) or 4294967295 (.*) (.*) (.*) (.*)",
    "ciod: In packet from node (.*) (.*) message still ready for node (.*) (.*) (.*) (.*) (.*)",
    "ciod: LOGIN chdir(.*) failed: Inputoutput error",
    "ciod: LOGIN chdir(.*) failed: No such file or directory",
    "ciod: LOGIN (.*) failed: Permission denied",
    "ciod: Message code (.*) is not (.*) or 4294967295",
    "ciod: Missing or invalid fields on line (.*) of node map file (.*)",
    "ciod: pollControlDescriptors: Detected the debugger died",
    "ciod: Received signal (.*) (.*) (.*) (.*)",
    "ciod: sendMsgToDebugger: error sending PROGRAM_EXITED message to debugger",
    "ciod: Unexpected eof at line (.*) of node map file (.*)",
    "ciodb has been restarted",
    "close EDRAM pages as soon as possible0",
    "comibmbgldevicesBulkPowerModule with VPD of comibmbgldevicesBulkPowerModuleVpdReply: IBM Part Number: 53P5763, Vendor: Cherokee International, Vendor Serial Number: 4274124, Assembly Revision:",
    "command manager unit summary0",
    "Controlling BGL rows  (.*)",
    "core configuration register: (.*)",
    "Core Configuration Register 0: (.*)",
    "correctable (.*)",
    "correctable error detected in directory (.*)",
    "correctable error detected in EDRAM bank (.*)",
    "critical input interrupt (.*)",
    "critical input interrupt (.*) (.*) warning for (.*) (.*) wire",
    "critical input interrupt (.*) (.*) warning for torus (.*) wire, suppressing further interrupts of same type",
    "data (.*) plb (.*)",
    "data address: (.*)",
    "data address space0",
    "data cache (.*) parity error detected attempting to correct",
    "data storage interrupt",
    "data store interrupt caused by (.*)",
    "data TLB error interrupt data address space0",
    "dbcr0=(.*) dbsr=(.*) ccr0=(.*)",
    "d-cache (.*) parity (.*)",
    "DCR (.*) : (.*)",
    "ddr: activating redundant bit steering for next allocation: (.*) (.*)",
    "ddr: activating redundant bit steering: (.*) (.*)",
    "ddr: excessive soft failures, consider replacing the card",
    "DDR failing (.*) register: (.*) (.*)",
    "DDR failing info register: (.*)",
    "DDR failing info register: DDR Fail Info Register: (.*)",
    "DDR machine check register: (.*) (.*)",
    "ddr: redundant bit steering failed, sequencer timeout",
    "ddr: Suppressing further CE interrupts",
    "ddr: Unable to steer (.*) (.*) - rank is already steering symbol (.*) Due to multiple symbols being over the correctabl[e]{0,1}",
    "ddr: Unable to steer (.*) (.*) - rank is already steering symbol (.*) Due to multiple symbols being over the correctable e",
    "ddr: Unable to steer (.*) (.*) - rank is already steering symbol (.*) Due to multiple symbols being over the correctable error threshold, consider replacing the card",
    "(.*) error threshold, consider replacing the card",
    "ddrSize == (.*)  ddrSize == (.*)",
    "debug interrupt enable0",
    "debug wait enable0",
    "DeclareServiceNetworkCharacteristics has been run but the DB is not empty",
    "DeclareServiceNetworkCharacteristics has been run with the force option but the DB is not empty",
    "disable all access to cache directory0",
    "disable apu instruction broadcast0",
    "disable flagging of DDR UE's as major internal error0",
    "disable speculative access0",
    "disable store gathering0",
    "disable trace broadcast0",
    "disable write lines 2:40",
    "divide-by-zero (.*)",
    "enable (.*) exceptions0",
    "enable invalid operation exceptions0",
    "enable non-IEEE mode0",
    "enabled exception summary0",
    "EndServiceAction (.*) performed upon (.*) by (.*)",
    "EndServiceAction (.*) was performed upon (.*) by (.*)",
    "EndServiceAction is restarting the (.*) cards in Midplane (.*) as part of Service Action (.*)",
    "EndServiceAction is restarting the (.*) in midplane (.*) as part of Service Action (.*)",
    "Error getting detailed hw info for node, caught javaioIOException: Problems with the chip, clear all resets",
    "Error getting detailed hw info for node, caught javaioIOException: Problems with the chip, could not enable clock domains",
    "Error getting detailed hw info for node, caught javaioIOException: Problems with the chip, could not pull all resets",
    "Error receiving packet on tree network, expecting type (.*) instead of type (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "Error receiving packet on tree network, packet index (.*) greater than max 366 (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "Error sending packet on tree network, packet at address (.*) is not aligned",
    "error threshold, consider replacing the card",
    "Error: unable to mount filesystem",
    "exception (.*)",
    "Exception Syndrome Register: (.*)",
    "exception syndrome register: (.*)",
    "Expected 10 active FanModules, but found 9  Found (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "external input interrupt (.*)",
    "external input interrupt (.*) (.*) (.*) tree receiver (.*) in resynch mode",
    "external input interrupt (.*) (.*) number of corrected SRAM errors has exceeded threshold",
    "external input interrupt (.*) (.*) number of corrected SRAM errors has exceeded threshold, suppressing further interrupts of same type",
    "external input interrupt (.*) (.*) torus sender (.*) retransmission error was corrected",
    "external input interrupt (.*) (.*) tree header with no target waiting",
    "external input interrupt (.*) (.*) uncorrectable torus error",
    "floating point (.*)",
    "floating point instr (.*)",
    "Floating Point Registers:",
    "Floating Point Status and Control Register: (.*)",
    "floating point unavailable interrupt",
    "floating pt ex mode (.*) (.*)",
    "force loadstore alignment0",
    "Found invalid node ecid in processor card slot (.*) ecid 0000000000000000000000000000",
    "fpr(.*)=(.*) (.*) (.*) (.*)",
    "fraction (.*)",
    "General Purpose Registers:",
    "general purpose registers:",
    "generating (.*)",
    "gister: machine state register: machine state register: machine state register: machine state register: machine state register:",
    "guaranteed (.*) cache block (.*)",
    "Hardware monitor caught javalangIllegalStateException: while executing I2C Operation caught javanetSocketException: Broken pipe and is stopping",
    "Hardware monitor caught javanetSocketException: Broken pipe and is stopping",
    "iar (.*) dear (.*)",
    "i-cache parity error0",
    "icache prefetch (.*)",
    "Ido chip status changed: (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "Ido packet timeout",
    "idoproxy communication failure: socket closed",
    "idoproxydb has been started: Name: (.*)  Input parameters: -enableflush -loguserinfo dbproperties BlueGene1",
    "idoproxydb hit ASSERT condition: ASSERT expression=(.*) Source file=(.*) Source line=(.*) Function=(.*) IdoTransportMgr::SendPacketIdoUdpMgr(.*), BglCtlPavTrace(.*)",
    "idoproxydb hit ASSERT condition: ASSERT expression=!nMsgLen > 0x10000 Source file=idomarshaleriocpp Source line=1929 Function=int IdoMarshalerRecvBuffer::ReadBlockIdoMsg::IdoMsgHdr(.*)&",
    "idoproxydb hit ASSERT condition: ASSERT expression=pTargetMgr Source file=idoclientmgrcpp Source line=353 Function=int IdoClientMgr::TargetCloseconst char(.*)",
    "idoproxydb hit ASSERT condition: ASSERT expression=!RecvMsgHdrulLen > 0x10000 Source file=idomarshaleriocpp Source line=387 Function=virtual int IdoMarshalerIo::RunRecv",
    "imprecise machine (.*)",
    "inexact (.*)",
    "0x[0-9a-fA-F]+ 0x[0-9a-fA-F]+",
    "instance of a correctable ddr RAS KERNEL INFO (.*) microseconds spent in the rbs signal handler during (.*) calls (.*) microseconds was the maximum time for a single instance of a correctable ddr",
    "instruction address: (.*)",
    "instruction address space0",
    "instruction cache parity error corrected",
    "instruction plb (.*)",
    "interrupt threshold0",
    "invalid (.*)",
    "invalid operation exception (.*)",
    "job (.*) timed out Block freed",
    "Kernel detected (.*) integer alignment exceptions (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*) (.*) iar (.*) dear (.*)",
    "kernel panic",
    "L1 DCACHE summary averages: #ofDirtyLines: (.*) out of 1024 #ofDirtyDblWord: (.*) out of 4096",
    "L3 (.*) (.*) register: (.*)",
    "L3 major internal error",
    "LinkCard is not fully functional",
    "lr:(.*) cr:(.*) xer:(.*) ctr:(.*)",
    "Lustre mount FAILED : (.*) : block_id : location",
    "Lustre mount FAILED : (.*) : point pgb1",
    "machine check (.*)",
    "MACHINE CHECK DCR read timeout (.*) iar (.*) lr (.*)",
    "machine check: i-fetch0",
    "machine check interrupt (.*) L2 dcache unit (.*) (.*) parity error",
    "machine check interrupt (.*) L2 DCU read error",
    "machine check interrupt (.*) L3 major internal error",
    "machine check interrupt (.*) TorusTreeGI read error 0",
    "machine check interrupt (.*) L2 dcache unit data parity error",
    "MACHINE CHECK PLB write IRQ (.*) iar (.*) lr (.*)",
    "Machine Check Status Register: (.*)",
    "machine check status register: (.*)",
    "machine state register:",
    "Machine State Register: (.*)",
    "machine state register: (.*)",
    "machine state register: machine state register: machine state register: machine state register: machine state register: machine",
    "MailboxMonitor::serviceMailboxes lib_ido_error: -1019 socket closed",
    "MailboxMonitor::serviceMailboxes lib_ido_error: -1114 unexpected socket error: Broken pipe",
    "mask(.*)",
    "max number of outstanding prefetches7",
    "max time in a cr RAS KERNEL INFO (.*) total interrupts (.*) critical input interrupts (.*) microseconds total spent on critical input interrupts, (.*) microseconds max time in a critical input interrupt",
    "memory and bus summary0",
    "memory manager (.*)",
    "memory manager  command manager address parity0",
    "memory manager address error0",
    "memory manager address parity error0",
    "memory manager refresh contention0",
    "memory manager refresh counter timeout0",
    "memory manager RMW buffer parity0",
    "memory manager store buffer parity0",
    "memory manager strobe gate0",
    "memory manager uncorrectable (.*)",
    "Microloader Assertion",
    "MidplaneSwitchController performing bit sparing on (.*) bit (.*)",
    "MidplaneSwitchController::clearPort bll_clear_port failed: (.*)",
    "MidplaneSwitchController::parityAlignment pap failed: (.*) (.*) (.*)",
    "MidplaneSwitchController::receiveTrain iap failed: (.*) (.*) (.*)",
    "MidplaneSwitchController::sendTrain port disconnected: (.*)",
    "minus (.*) (.*)",
    "minus (.*)",
    "miscompare0",
    "Missing reverse cable: Cable (.*) (.*) (.*) (.*) --> (.*) (.*) (.*) (.*) is present BUT the reverse cable (.*) (.*) (.*) (.*) --> (.*) (.*) (.*) (.*) is missing",
    "mmcs_db_server has been started: bglBlueLightppcfloorbglsysbinmmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties dbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties dbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --shutdown-timeout 120",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties dbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --shutdown-timeout 120 --shutdown-timeout 240",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --no-reconnect-blocks",
    "mmcs_db_server has been started: mmcs_db_server --useDatabase BGL --dbproperties serverdbproperties --iolog bglBlueLightlogsBGL --reconnect-blocks all --shutdown-timeout (.*)",
    "mmcs_server exited abnormally due to signal: Segmentation fault",
    "monitor caught javalangIllegalStateException: while executing CONTROL Operation caught javaioEOFException and is stopping",
    "monitor caught javalangIllegalStateException: while executing (.*) Operation caught javanetSocketException: Broken pipe and is stopping",
    "monitor caught javalangUnsupportedOperationException: power module (.*) not present and is stopping",
    "msr=(.*) dear=(.*) esr=(.*) fpscr=(.*)",
    "New ido chip inserted into the database: (.*) (.*) (.*) (.*)",
    "NFS Mount failed on (.*) slept (.*) seconds, retrying (.*)",
    "no ethernet link",
    "No power module (.*) found found on link card",
    "Node card is not fully functional",
    "Node card status: ALERT 0, ALERT 1, ALERT 2, ALERT 3 is are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted TEMPERATURE MASK IS ACTIVE No temperature error Temperature Limit Error Latch is clear PGOOD is asserted PGOOD error latch is clear MPGOOD is OK MPGOOD error latch is clear The 25 volt rail is OK The 15 volt rail is OK",
    "Node card status: ALERT 0, ALERT 1, ALERT 2, ALERT 3 is are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD is asserted PGOOD error latch is clear MPGOOD is OK MPGOOD error latch is clear The 25 volt rail is OK The 15 volt rail is OK",
    "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD IS NOT OK MPGOOD ERROR LATCH IS ACTIVE THE 25 VOLT RAIL IS NOT OK THE 15 VOLT RAIL IS NOT OK",
    "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD IS NOT OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
    "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD is OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
    "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD is OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
    "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD IS NOT OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
    "Node card status: no ALERTs are active Clock Mode is Low Clock Select is Midplane Phy JTAG Reset is asserted ASIC JTAG Reset is not asserted Temperature Mask is not active No temperature error Temperature Limit Error Latch is clear PGOOD IS NOT ASSERTED PGOOD ERROR LATCH IS ACTIVE MPGOOD is OK MPGOOD ERROR LATCH IS ACTIVE The 25 volt rail is OK The 15 volt rail is OK",
    "Node card VPD check: (.*) node in processor card slot (.*) do not match VPD ecid (.*) found (.*)",
    "Node card VPD check: missing (.*) node, VPD ecid (.*) in processor card slot (.*)",
    "NodeCard is not fully functional",
    "NodeCard temperature sensor chip (.*) is not accessible",
    "NodeCard VPD chip is not accessible",
    "NodeCard VPD is corrupt",
    "number of correctable errors detected in L3 (.*)",
    "number of lines with parity errors written to L3 (.*)",
    "overflow exception0",
    "parity error in bank (.*)",
    "parity error in read queue (.*)",
    "parity error in write buffer0",
    "parity error0",
    "plus (.*)",
    "Power deactivated: (.*)",
    "Power Good signal deactivated: (.*) A service action may be required",
    "power module status fault detected on node card status registers are: (.*)",
    "prefetch depth for core (.*)",
    "prefetch depth for PLB slave1",
    "PrepareForService is being done on this (.*) (.*) (.*) (.*) (.*) by (.*)",
    "PrepareForService is being done on this Midplane (.*) (.*) (.*) by (.*)",
    "PrepareForService is being done on this part (.*) (.*) (.*) (.*) (.*) by (.*)",
    "PrepareForService is being done on this rack (.*) by (.*)",
    "PrepareForService shutting down (.*) as part of Service Action (.*)",
    "Problem communicating with link card iDo machine with LP of (.*) caught javalangIllegalStateException: while executing I2C Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 13 and card type 1 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = 845, Reply Sequence Number = -1, timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = -1, Expected Sync Command = 32, Actual Sync Reply = -1",
    "Problem communicating with node card, iDo machine with LP of (.*) caught javalangIllegalStateException: while executing (.*) Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 13 and card type 4 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = 0, Reply Sequence Number = -1, timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = -1, Expected Sync Command = 8, Actual Sync Reply = -1",
    "Problem communicating with service card, ido chip: (.*) javaioIOException: Could not find EthernetSwitch on port:address (.*)",
    "Problem communicating with service card, ido chip: (.*) javalangIllegalStateException: IDo is not in functional state -- currently in state COMMUNICATION_ERROR",
    "Problem communicating with service card, ido chip: (.*) javalangIllegalStateException: while executing CONTROL Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 9 and card type 2 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = (.*) Reply Sequence Number = (.*) timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = (.*) Expected Sync Command = 8, Actual Sync Reply = (.*)",
    "Problem reading the ethernet arl entries fro the service card: javalangIllegalStateException: while executing I2C Operation caught javalangRuntimeException: Communication error: DirectIDo for comibmidoDirectIDo object (.*) with image version 9 and card type 2 is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = (.*) Reply Sequence Number = (.*) timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 2, Actual Op Reply = (.*) Expected Sync Command = 32, Actual Sync Reply = (.*)",
    "problem state (.*)",
    "program interrupt",
    "program interrupt: fp compare0",
    "program interrupt: fp cr (.*)",
    "program interrupt: illegal (.*)",
    "program interrupt: imprecise exception0",
    "program interrupt: privileged instruction0",
    "program interrupt: trap (.*)",
    "program interrupt: unimplemented operation0",
    "quiet NaN0",
    "qw trapped0",
    "r(.*)=(.*) r(.*)=(.*) r(.*)=(.*) r(.*)=(.*)",
    "regctl scancom interface0",
    "reserved0",
    "round nearest0",
    "round toward (.*)",
    "rts assertion failed: `personality->version == BGLPERSONALITY_VERSION' in `void start' at startcc:131",
    "rts assertion failed: `vaddr % PAGE_SIZE_1M == 0' in `int initializeAppMemoryint, TLBEntry&, unsigned int, unsigned int' at mmucc:540",
    "rts: bad message header: cpu (.*) invalid (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "rts: bad message header: expecting type (.*) instead of type (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "rts: bad message header: index 0 greater than total 0 (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "rts: bad message header: packet index (.*) greater than max 366 (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "rts internal error",
    "rts: kernel terminated for reason (.*)",
    "rts: kernel terminated for reason 1001rts: bad message header: invalid cpu, (.*) (.*) (.*) (.*)",
    "rts: kernel terminated for reason 1002rts: bad message header: too many packets, (.*) (.*) (.*) (.*) (.*)",
    "rts: kernel terminated for reason 1004rts: bad message header: expecting type (.*) (.*) (.*) (.*) (.*)",
    "rts panic! - stopping execution",
    "rts treetorus link training failed: wanted: (.*) got: (.*)",
    "Running as background command",
    "(.*) (.*) StatusA",
    "shutdown complete",
    "size of DDR we are caching1 512M",
    "size of scratchpad portion of L30 0M",
    "Special Purpose Registers:",
    "special purpose registers:",
    "start (.*)",
    "Starting SystemController",
    "state machine0",
    "state register: machine state register: machine state register: machine state register: machine state register: machine state re",
    "store (.*)",
    "summary1",
    "suppressing further interrupts of same type",
    "symbol(.*)",
    "Target=(.*) Message=(.*)",
    "(.*) (.*) All all zeros, power good may be low",
    "(.*) (.*) failed to lock",
    "(.*) (.*) JtagId = (.*)",
    "(.*) (.*) JtagId = (.*) Run environmental monitor to diagnose possible hardware failure",
    "Temperature Over Limit on link card",
    "this link card is not fully functional",
    "tlb (.*)",
    "Torus non-recoverable error DCRs follow",
    "total of (.*) ddr errors detected and corrected",
    "total of (.*) ddr errors detected and corrected over (.*) seconds",
    "turn on hidden refreshes1",
    "uncorrectable (.*)",
    "uncorrectable error detected in (.*) (.*)",
    "uncorrectable error detected in EDRAM bank (.*)",
    "underflow (.*)",
    "VALIDATE_LOAD_IMAGE_CRC_IN_DRAM",
    "wait state enable0",
    "While initializing link card iDo machine with LP of (.*) caught javaioIOException: Could not contact iDo with (.*) and (.*) because javalangRuntimeException: Communication error: (.*)",
    "While initializing node card, ido with LP of (.*) caught javalangIllegalStateException: IDo is not in functional state -- currently in state COMMUNICATION_ERROR",
    "While initializing node card, ido with LP of (.*) caught javalangIllegalStateException: while executing CONTROL Operation caught javalangRuntimeException: Communication error: (.*)",
    "While initializing node card, ido with LP of (.*) caught javalangNullPointerException",
    "While initializing node card, ido with LP of (.*) caught javalangIllegalStateException: while executing I2C Operation caught javalangRuntimeException: Communication error: (.*)",
    "While initializing (.*) card iDo with LP of (.*) caught javaioIOException: Could not contact iDo with LP=(.*) and IP=(.*) because javalangRuntimeException: Communication error: DirectIDo for Uninitialized DirectIDo for (.*) is in state = COMMUNICATION_ERROR, sequenceNumberIsOk = false, ExpectedSequenceNumber = 0, Reply Sequence Number = -1, timedOut = true, retries = 200, timeout = 1000, Expected Op Command = 5, Actual Op Reply = -1, Expected Sync Command = 10, Actual Sync Reply = -1",
    "While inserting monitor info into DB caught javalangNullPointerException",
    "While reading FanModule caught javalangIllegalStateException: while executing I2C Operation caught (.*)",
    "While reading FanModule caught javalangIllegalStateException: while executing I2C Operation caught javanetSocketException: Broken pipe",
    "While setting fan speed caught javalangIllegalStateException: while executing I2C Operation caught (.*)",
    "write buffer commit threshold2",
    "DDR failing data registers: (.*) (.*)",
    "program interrupt: fp cr field 0",
    "rts treetorus link training failed: wanted: (.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*) got: (.*) (.*) (.*) (.*) (.*) (.*) (.*)",
    "rts: kernel terminated for reason 1002rts: bad message header: too many packets, (.*) (.*) (.*) (.*)",
    "(.*)",
]
