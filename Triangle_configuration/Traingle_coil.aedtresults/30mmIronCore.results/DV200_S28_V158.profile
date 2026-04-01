$begin 'Profile'
	$begin 'ProfileGroup'
		MajorVer=2025
		MinorVer=2
		Name='Solution Process'
		$begin 'StartInfo'
			I(1, 'Start Time', '03/31/2026 11:10:09')
			I(1, 'Host', 'LAPTOP-FPHQ2IVC')
			I(1, 'Processor', '16')
			I(1, 'OS', 'NT 10.0')
			I(1, 'Product', 'Maxwell 3D 2025.2.0')
		$end 'StartInfo'
		$begin 'TotalInfo'
			I(1, 'Elapsed Time', '00:00:43')
			I(1, 'ComEngine Memory', '70 M')
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
		ProfileItem('Design Validation', 0, 0, 0, 0, 0, 'I(3, 1, \'Level\', \'Perform full validations\', 1, \'Elapsed Time\', \'00:00:00\', 2, \'Memory\', 71316, true)', false, true)
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=2
			Name='Initial Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '03/31/2026 11:10:09')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:05')
			$end 'TotalInfo'
			GroupOptions=4
			TaskDataOptions(Memory=8)
			ProfileItem('Mesh', 1, 0, 2, 0, 59000, 'I(3, 2, \'Tetrahedra\', 18088, false, 1, \'Type\', \'TAU\', 2, \'Cores\', 4, false)', true, true)
			ProfileItem('Coarsen', 1, 0, 1, 0, 59000, 'I(1, 2, \'Tetrahedra\', 4792, false)', true, true)
			ProfileItem('Post', 2, 0, 2, 0, 60896, 'I(1, 2, \'Tetrahedra\', 4691, false)', true, true)
		$end 'ProfileGroup'
		$begin 'ProfileGroup'
			MajorVer=2025
			MinorVer=2
			Name='Adaptive Meshing'
			$begin 'StartInfo'
				I(1, 'Time', '03/31/2026 11:10:15')
			$end 'StartInfo'
			$begin 'TotalInfo'
				I(1, 'Elapsed Time', '00:00:38')
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
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 13387, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 6333, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 2, 0, 2, 0, 188776, 'I(1, 2, \'Tetrahedra\', 4691, false)', true, true)
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
				ProfileItem('Adaptive Refine', 1, 0, 1, 0, 35436, 'I(1, 2, \'Tetrahedra\', 6104, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 18682, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 8406, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 2, 0, 3, 0, 199600, 'I(1, 2, \'Tetrahedra\', 6104, false)', true, true)
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
				ProfileItem('Adaptive Refine', 2, 0, 2, 0, 38892, 'I(1, 2, \'Tetrahedra\', 7938, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 0, 0, 27960, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 10999, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 3, 0, 3, 0, 216296, 'I(1, 2, \'Tetrahedra\', 7938, false)', true, true)
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
				ProfileItem('Adaptive Refine', 2, 0, 2, 0, 41396, 'I(1, 2, \'Tetrahedra\', 10327, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 40736, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 14346, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 3, 0, 3, 0, 230924, 'I(1, 2, \'Tetrahedra\', 10327, false)', true, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=2
				Name='Pass 5'
				$begin 'StartInfo'
					I(0, '')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions(Memory=8)
				ProfileItem('Adaptive Refine', 2, 0, 2, 0, 44948, 'I(1, 2, \'Tetrahedra\', 13435, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 57356, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 18602, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 4, 0, 4, 0, 249124, 'I(1, 2, \'Tetrahedra\', 13435, false)', true, true)
			$end 'ProfileGroup'
			ProfileItem('', 0, 0, 0, 0, 0, 'I(1, 0, \'\')', false, true)
			$begin 'ProfileGroup'
				MajorVer=2025
				MinorVer=2
				Name='Pass 6'
				$begin 'StartInfo'
					I(0, '')
				$end 'StartInfo'
				$begin 'TotalInfo'
					I(0, ' ')
				$end 'TotalInfo'
				GroupOptions=0
				TaskDataOptions(Memory=8)
				ProfileItem('Adaptive Refine', 3, 0, 3, 0, 48096, 'I(1, 2, \'Tetrahedra\', 17472, false)', true, true)
				ProfileItem('Matrix Solve', 0, 0, 1, 0, 76280, 'I(4, 1, \'Type\', \'DRS\', 2, \'Cores\', 4, false, 2, \'Matrix\', 24072, false, 1, \'Disk\', \'0KB\')', true, true)
				ProfileItem('adapt', 3, 0, 5, 0, 270228, 'I(1, 2, \'Tetrahedra\', 17472, false)', true, true)
			$end 'ProfileGroup'
			ProfileFootnote('I(1, 0, \'Adaptive Passes converged\')', 0)
		$end 'ProfileGroup'
		ProfileFootnote('I(2, 1, \'Stop Time\', \'03/31/2026 11:10:53\', 1, \'Status\', \'Normal Completion\')', 0)
	$end 'ProfileGroup'
$end 'Profile'
