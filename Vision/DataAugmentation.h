#pragma once

enum AugmentationType {
	SCALING,
	TRANSLATION,
	ROTATION,
	FINER_ROTATION,
	FLIPPING,
	PEPPER_AND_SALT,
	LIGHTNING,
	PERSPECTIVE,
};
class DataAugmentation
{

public:
	DataAugmentation();
	~DataAugmentation();
};

