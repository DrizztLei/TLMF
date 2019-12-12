package cluster;

public class Node
{
	private int id;
	private int length;
	private double cov;
	private String sequence;

	public Node()
	{

	}

	public int getId()
	{
		return id;
	}

	public void setId(int id)
	{
		this.id = id;
	}

	public int getLength()
	{
		return length;
	}

	public void setLength(int length)
	{
		this.length = length;
	}

	public double getCov()
	{
		return cov;
	}

	public void setCov(double cov)
	{
		this.cov = cov;
	}

	public String getSequence()
	{
		return sequence;
	}

	public void setSequence(String sequence)
	{
		this.sequence = sequence;
	}
}
