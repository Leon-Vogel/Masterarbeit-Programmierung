DROP TABLE IF EXISTS SubstationTaskInfo, OrderAssignment, Shift, SubstationError, SetupTimes, Maintenance, BOM,
 OrdersPredecessor, Pause, EquipmentAssignment, EquipmentPlan, EquipmentMaterial, Equipment,  MTMTime, Substation,  SimOrders, StationTaskInfo, Station;

 CREATE TABLE Station (
	StationID INTEGER PRIMARY KEY,
	Name VARCHAR(20),
	Type VARCHAR(20),
	MaxSlots INTEGER,
	SetupKey INTEGER,
	maxEmployees INTEGER,
	minEmployees INTEGER,
	nEmployees INTEGER,
	currSpeed DECIMAL(10, 5)
	);

CREATE TABLE Substation (
	SubstationID INTEGER PRIMARY KEY,
	Name VARCHAR(60),
	Type VARCHAR(20),
	StationID INTEGER,
	Sequence INTEGER,
	FOREIGN KEY (StationID) REFERENCES Station(StationID)
	);

CREATE TABLE MTMTime (
    SubstationID Integer,
    Material VARCHAR(20),
    Time DECIMAL(15, 5),
    PRIMARY KEY (SubstationID, Material),
    FOREIGN KEY (SubstationID) REFERENCES Substation(SubstationID)
    );

CREATE TABLE SubstationError (
	SubstationID INTEGER,
	ErrorCode INTEGER,
	ErroDescription VARCHAR(20),
	Seconds DECIMAL(15, 5),
	Prob DECIMAL(15, 5),
	PRIMARY KEY (SubstationID, ErrorCode),
	FOREIGN KEY (SubstationID) REFERENCES Substation(SubstationID)
	);

CREATE TABLE SetupTimes (
	StationID INTEGER,
	FromType VARCHAR(25),
	ToType VARCHAR(25),
	Time DECIMAL(15, 5),
	PRIMARY KEY (StationID, FromType, ToType),
	FOREIGN KEY (StationID) REFERENCES Station(StationID)
	);

CREATE TABLE SimOrders (
    OrderID INTEGER PRIMARY KEY,
    Amount INTEGER,
    DueDate VARCHAR(20),
    EarliestDueDate VARCHAR(20),
    Clockrate FLOAT,
	SetupKey VARCHAR(30),
    Accomp INTEGER,
    Running INTEGER
    );

CREATE TABLE OrdersPredecessor (
    OrderPredecessorID INTEGER,
    OrderSuccessorID INTEGER,
    MaximumTimeBetween INTEGER,
    PRIMARY KEY (OrderPredecessorID, OrderSuccessorID),
    FOREIGN KEY (OrderPredecessorID) REFERENCES SimOrders(OrderID),
    FOREIGN KEY (OrderSuccessorID) REFERENCES SimOrders(OrderID)
    );

CREATE TABLE OrderAssignment (
    OrderID INTEGER,
    StationID INTEGER,
    PlannedStart DATETIME,
    PlannedEnd DATETIME,
    TimePassed INTEGER,
    n_Employee INTEGER,
    FOREIGN KEY (StationID) REFERENCES Station(StationID),
    FOREIGN KEY (OrderID) REFERENCES SimOrders(OrderID)
    );

CREATE TABLE Equipment (
    EquipmentID INTEGER PRIMARY KEY,
    Type INTEGER,
    Name VARCHAR(60),
    );

CREATE TABLE EquipmentAssignment (
    EquipmentID INTEGER,
    OrderID INTEGER,
    StationID INTEGER,
    PRIMARY KEY (EquipmentID, StationID),
    FOREIGN KEY (StationID) REFERENCES Station(StationID),
    FOREIGN KEY (OrderID) REFERENCES SimOrders(OrderID),
    FOREIGN KEY (EquipmentID) REFERENCES Equipment(EquipmentID)
    );

CREATE TABLE EquipmentPlan (
    EquipmentID INTEGER,
    StartPause DATETIME,
    EndPause DATETIME,
    FOREIGN KEY (EquipmentID) REFERENCES Equipment(EquipmentID)
    );

CREATE TABLE EquipmentMaterial (
    Type INTEGER,
    Material INTEGER,
	StationID INTEGER,
	FOREIGN KEY (StationID) REFERENCES Station(StationID)
    );

CREATE TABLE Pause (
    StationID INTEGER,
    MaintenanceStart DATETIME,
    MaintenanceEnd DATETIME,
    PRIMARY KEY (StationID, MaintenanceStart, MaintenanceEnd),
    FOREIGN KEY (StationID) REFERENCES Station(StationID)
    );

CREATE TABLE Shift (
    ShiftID INTEGER,
    ShiftStart DATETIME,
    ShiftEnd DATETIME,
    ShiftDay INTEGER,
    n_Employee INTEGER
);

CREATE TABLE SubstationTaskInfo (
	SubstationID INTEGER,
	TaskID INTEGER,
	avgTime DECIMAL(15, 5),
	varTime DECIMAL(15, 5),
    Clockrate DECIMAL(15, 5),
    EquipmentID INTEGER,
    nEmployees INTEGER,
    Slots INTEGER,
	ReworkProb DECIMAL(15, 5),
	samples INTEGER,
	RepStationID INTEGER,
	FOREIGN KEY (SubstationID) REFERENCES Substation(SubstationID),
	);

CREATE TABLE BOM (
    OrderID INTEGER,
    Material INTEGER,
    MaterialName Varchar(50),
    Position INTEGER,
    FOREIGN KEY (OrderID) REFERENCES SimOrders(OrderID),
    PRIMARY KEY (OrderID, Material)
);


