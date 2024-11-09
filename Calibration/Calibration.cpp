#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

using namespace std;

typedef union
{
	struct
	{
		float x;
		float y;
	};
	struct
	{
		float i;
		float j;
	};
	struct
	{
		float a;
		float b;
	};
	struct
	{
		float w;
		float h;
	};
} Vec2d;
typedef union
{
	struct
	{
		float x;
		float y;
		float z;
	};
	struct
	{
		float i;
		float j;
		float k;
	};
	struct
	{
		float a;
		float b;
		float c;
	};
} Vec3d;

inline float Deg2Rad(float degree)
{
	return degree * (EIGEN_PI / 180.0f);
}
inline Vec3d Deg2Rad(Vec3d xyz)
{
	return Vec3d{ Deg2Rad(xyz.x), Deg2Rad(xyz.y), Deg2Rad(xyz.z) };
}

Eigen::Matrix4f GetXRotationMatrix(const float radian)
{
	const float sinTheta = sin(radian);
	const float cosTheta = cos(radian);

	Eigen::Matrix4f m_RotationX;
	m_RotationX <<
		1, 0, 0, 0,
		0, cosTheta, -sinTheta, 0,
		0, sinTheta, cosTheta, 0,
		0, 0, 0, 1;

	return m_RotationX;
}
Eigen::Matrix4f GetYRotationMatrix(const float radian)
{
	const float sinTheta = sin(radian);
	const float cosTheta = cos(radian);

	Eigen::Matrix4f m_RotationY;
	m_RotationY <<
		cosTheta, 0, sinTheta, 0,
		0, 1, 0, 0,
		-sinTheta, 0, cosTheta, 0,
		0, 0, 0, 1;

	return m_RotationY;
}
Eigen::Matrix4f GetZRotationMatrix(const float radian)
{
	const float sinTheta = sin(radian);
	const float cosTheta = cos(radian);

	Eigen::Matrix4f m_RotationZ;
	m_RotationZ <<
		cosTheta, -sinTheta, 0, 0,
		sinTheta, cosTheta, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	return m_RotationZ;
}
Eigen::Matrix4f GetRotationMatrix(const Vec3d xyz)
{
	const Eigen::Matrix4f m_RotationX = GetXRotationMatrix(xyz.x);
	const Eigen::Matrix4f m_RotationY = GetYRotationMatrix(xyz.y);
	const Eigen::Matrix4f m_RotationZ = GetZRotationMatrix(xyz.z);

	const Eigen::Matrix4f m_Rotation = m_RotationZ * m_RotationY * m_RotationX;

	return m_Rotation;
}

Eigen::Matrix4f GetTranslationMatrix(const Vec3d xyz)
{
	Eigen::Matrix4f m_Translation = Eigen::Matrix4f::Identity();
	m_Translation(0, 3) = xyz.x;
	m_Translation(1, 3) = xyz.y;
	m_Translation(2, 3) = xyz.z;

	return m_Translation;
}

Eigen::Matrix4f GetExtrinsicMatrix(const Vec3d rotation1, const Vec3d translation, const Vec3d rotation2)
{
	const Eigen::Matrix4f m_Rotation1 = GetRotationMatrix(rotation1);
	const Eigen::Matrix4f m_Translation = GetTranslationMatrix(translation);
	const Eigen::Matrix4f m_Rotation2 = GetRotationMatrix(rotation2);
	const Eigen::Matrix4f m_Extrinsic = m_Rotation2 * m_Translation * m_Rotation1;

	return m_Extrinsic;
}

Eigen::Matrix4f GetExpandMatrix(const float imageHeight)
{
	Eigen::Matrix4f m_Expand = Eigen::Matrix4f::Identity();
	m_Expand(0, 0) = m_Expand(1, 1) = imageHeight / 2.0f;

	return m_Expand;
}
Eigen::Matrix4f GetIntrinsicMatrix(const float fov, const float aspect)
{
	const float focalLength = sqrt(1.0f + aspect * aspect) / tan(fov / 2.0f);

	Eigen::Matrix4f m_Intrinsic = Eigen::Matrix4f::Identity();
	m_Intrinsic(0, 0) = m_Intrinsic(1, 1) = focalLength;

	return m_Intrinsic;
}

Eigen::Matrix4f GetTransformationMatrix(const Vec2d imageShape,
	const Vec3d rotation1, const Vec3d translation, const Vec3d rotation2, const float fov)
{
	const Eigen::Matrix4f m_Extrinsic = GetExtrinsicMatrix(rotation1, translation, rotation2);
	const Eigen::Matrix4f m_Intrinsic = GetIntrinsicMatrix(fov, imageShape.w / imageShape.h);
	const Eigen::Matrix4f m_Expand = GetExpandMatrix(imageShape.h);
	const Eigen::Matrix4f m_Transformation = m_Expand * m_Intrinsic * m_Extrinsic;

	return m_Transformation;
}

pair<Eigen::MatrixX4f, vector<int>> ProjectPoints(const Eigen::MatrixX3f pointCloud, const Eigen::Matrix4f m_Transformation,
	const Vec2d imageShape)
{
	Eigen::MatrixX4f pointCloudHomogeneous(pointCloud.rows(), 4);
	pointCloudHomogeneous << pointCloud, Eigen::VectorXf::Ones(pointCloud.rows());

	Eigen::MatrixX4f pointCloudTrans = pointCloudHomogeneous * m_Transformation;
	pointCloudTrans.col(0) = pointCloudTrans.col(0).array() / pointCloudTrans.col(2).array() + imageShape.w / 2.0f;
	pointCloudTrans.col(1) = pointCloudTrans.col(1).array() / pointCloudTrans.col(2).array() + imageShape.h / 2.0f;

	std::vector<int> indicesInFov;
	for (int i = 0; i < pointCloudTrans.rows(); ++i)
	{
		Eigen::Vector4f row = pointCloudTrans.row(i);
		bool isInWidth = row.x() >= 0.0f && row.x() < imageShape.w;
		bool isInHeight = row.y() >= 0.0f && row.y() < imageShape.h;
		bool isFront = row.z() > 0.0f;

		if (isInWidth && isInHeight && isFront)
		{
			indicesInFov.push_back(i);
		}
	}

	Eigen::MatrixX4f filteredPointCloud(indicesInFov.size(), 4);
	for (int i = 0; i < indicesInFov.size(); ++i)
	{
		filteredPointCloud.row(i) = pointCloudTrans.row(indicesInFov[i]);
	}

	return pair<Eigen::MatrixX4f, vector<int>>(filteredPointCloud, indicesInFov);
}


int main(void)
{
	const float fov = Deg2Rad(90);
	const Vec2d imageShape = { 640, 480 };
	const Vec3d rotation1Degrees = { 90.0f, -90.0f, 0.0f };
	const Vec3d translation = { 0.0f, 0.0f, 3.0f };
	const Vec3d rotation2Degrees = { 0.0f, 0.0f, 0.0f };

	const Vec3d rotation1 = Deg2Rad(rotation1Degrees);
	const Vec3d rotation2 = Deg2Rad(rotation2Degrees);

	const Eigen::Matrix4f m_Transformation = GetTransformationMatrix(imageShape, rotation1, translation, rotation2, fov).transpose();

	Eigen::MatrixX3f pointCloud(6, 3);
	pointCloud.row(0) = Eigen::Vector3f(1, 1, 1);
	pointCloud.row(1) = Eigen::Vector3f(1, 0, 0);
	pointCloud.row(2) = Eigen::Vector3f(0, 1, 0);
	pointCloud.row(3) = Eigen::Vector3f(0, 0, 1);
	pointCloud.row(4) = Eigen::Vector3f(1, 2, 3);
	pointCloud.row(5) = Eigen::Vector3f(0, 0, 0);

	pair<Eigen::MatrixX4f, vector<int>> result = ProjectPoints(pointCloud, m_Transformation, imageShape);
	Eigen::MatrixX4f projectedPoints = result.first;
	vector<int> indices = result.second;

	cout << projectedPoints << endl;

	return 0;
}