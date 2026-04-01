$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=2
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '03/26/2026 14:47:46')
			I(1, 'Host', 'LAPTOP-FPHQ2IVC')
			I(1, 'Processor', '16')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'Maxwell 3D 2025.2.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:15')
			I(1, 'ComEngine Memory', '73 M')
		$end 'TotalInfo'
		GroupOptions=8
		TaskDataOptions(Memory=8)
		ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 1, \'Executing From\', \'C:\\\\Program Files\\\\ANSYS Inc\\\\ANSYS Student\\\\v252\\\\AnsysEM\\\\MAXWELLCOMENGINE.exe\')', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=2
			Name='HPC'
			$begin 'StartInfo'
				I(1, 'Type', 'Manual')
				I(1, 'Distribution Types', 'Variations, Frequencies')
				I(1, 'MPI Vendor', 'Intel')
				I(1, 'MPI Version', '2021')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(0, ' ')
			$end 'TotalInfo'
			GroupOptions=0
			TaskDataOptions(Memory=8)
			ProfileItem('Machine', 0, 0, 0, 0, 0, 'I(4, 1, \'Name\', \'LAPTOP-FPHQ2IVC\', 3, \'RAM Limit\', 90, \'%f%%\', 2, \'Tasks\', 1, false, 2, \'Cores\', 4, false)', false, true)
		$end 'ProfileGroup'
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(3, 1, \'Level\', \'Perform full validations\', 1, \'Elapsed Time\', \'00:00:00\', 2, \'Memory\', 74720, true)', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=2
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '03/26/2026 14:47:46')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:03')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions(Memory=8)
			ProfileItem('Mesh', 1, 0, 2, 0, 64000, 'I(3, 2, \'Tetrahedra\', 22314, false, 1, \'Type\', \'TAU\', 2, \'Cores\', 4, false)', true, true)
			ProfileItem('Coarsen', 1, 0, 1, 0, 64000, 'I(1, 2, \'Tetrahedra\', 5613, false)', true, true)
			ProfileItem('Post', 2, 0, 1, 0, 65944, 'I(1, 2, \'Tetrahedra\', 5612, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=2
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '03/26/2026 14:47:50')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:12')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions(Memory=8)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=2
				Name='Pass 1'
				$begin 'StartInfo'
					I(0, '')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions(Memory=8)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 15485, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 7631, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 0, 0, 2, 0, 196920, 'I(1, 2, \'Tetrahedra\', 5612, false)', true, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=2
				Name='Pass 2'
				$begin 'StartInfo'
					I(0, '')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions(Memory=8)
				ProfileItem('Adaptive Refine', 0, 0, 0, 0, 37188, 'I(1, 2, \'Tetrahedra\', 7299, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 22725, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 10116, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 0, 0, 2, 0, 208868, 'I(1, 2, \'Tetrahedra\', 7299, false)', true, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=2
				Name='Pass 3'
				$begin 'StartInfo'
					I(0, '')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions(Memory=8)
				ProfileItem('Adaptive Refine', 0, 0, 1, 0, 40040, 'I(1, 2, \'Tetrahedra\', 9493, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 34504, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 13248, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 2, 0, 2, 0, 224836, 'I(1, 2, \'Tetrahedra\', 9493, false)', true, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=2
				Name='Pass 4'
				$begin 'StartInfo'
					I(0, '')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions(Memory=8)
				ProfileItem('Adaptive Refine', 1, 0, 1, 0, 43356, 'I(1, 2, \'Tetrahedra\', 12349, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 50184, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 17218, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 2, 0, 3, 0, 240344, 'I(1, 2, \'Tetrahedra\', 12349, false)', true, true)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'03/26/2026 14:48:02\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
